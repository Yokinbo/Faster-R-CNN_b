import json
import os

import matplotlib
import torch

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import shutil
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .utils import cvtColor, resize_image, preprocess_input, get_new_img_size
from .utils_bbox import DecodeBox
from .utils_map import get_coco_map

class LossHistory():
    def __init__(self, log_dir, model, input_shape):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        # try:
        #     dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
        #     self.writer.add_graph(model, dummy_input)
        # except:
        #     pass

    def append_loss(self, epoch, loss, val_loss):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")

        self.writer.add_scalar('loss', loss, epoch)
        self.writer.add_scalar('val_loss', val_loss, epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, class_names, num_classes, val_lines, log_dir, cuda, \
            map_out_path=".temp_map_out", max_boxes=100, confidence=0.05, nms_iou=0.5, letterbox_image=True, MINOVERLAP=0.5, eval_flag=True, period=1, annotation_json=None):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.class_names        = class_names
        self.num_classes        = num_classes
        self.val_lines          = val_lines
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.map_out_path       = map_out_path
        self.max_boxes          = max_boxes
        self.confidence         = confidence
        self.nms_iou            = nms_iou
        self.letterbox_image    = letterbox_image
        self.MINOVERLAP         = MINOVERLAP
        self.eval_flag          = eval_flag
        self.period             = period
        self.annotation_json    = annotation_json
        self.val_ground_truth   = self._load_coco_ground_truth(annotation_json)
        
        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        if self.cuda:
            self.std    = self.std.cuda()
        self.bbox_util  = DecodeBox(self.std, self.num_classes)

        self.maps       = [0]
        self.epoches    = [0]
        self.best_map50 = float("-inf")
        self.best_map50_epoch = -1
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def _load_coco_ground_truth(self, annotation_json):
        """读取验证集原始 COCO 浮点框，供训练期 AP50 选权重使用。"""
        if annotation_json is None:
            raise ValueError("训练期验证必须提供 val.json，禁止使用取整后的临时真值框。")
        with open(annotation_json, encoding="utf-8") as file:
            payload = json.load(file)
        category_to_index = {}
        for category in payload.get("categories", []):
            name = category["name"]
            if name not in self.class_names:
                raise ValueError(f"val.json 类别 {name!r} 不在 classes.txt 中。")
            category_to_index[int(category["id"])] = self.class_names.index(name)
        annotations_by_image = {}
        for annotation in payload.get("annotations", []):
            annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        ground_truth = {}
        for image in payload.get("images", []):
            boxes = []
            for annotation in annotations_by_image.get(int(image["id"]), []):
                left, top, width, height = map(float, annotation["bbox"])
                boxes.append(
                    [
                        left,
                        top,
                        left + width,
                        top + height,
                        category_to_index[int(annotation["category_id"])],
                    ]
                )
            ground_truth[image["file_name"].casefold()] = boxes
        if not ground_truth:
            raise ValueError(f"val.json 没有图像记录：{annotation_json}")
        return ground_truth

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w")
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # Use the same fixed input size as training and standalone validation.
        input_shape = self.input_shape
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        
        #---------------------------------------------------------#
        #   给原图像进行resize，resize到短边为600的大小上
        #---------------------------------------------------------#
        image_data  = resize_image(image, [input_shape[1], input_shape[0]])
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            roi_cls_locs, roi_scores, rois, _ = self.net(images)
            #-------------------------------------------------------------#
            #   利用classifier的预测结果对建议框进行解码，获得预测框
            #-------------------------------------------------------------#
            results = self.bbox_util.forward(roi_cls_locs, roi_scores, rois, image_shape, input_shape, 
                                                    nms_iou = self.nms_iou, confidence = self.confidence)
            #--------------------------------------#
            #   如果没有检测到物体，则返回原图
            #--------------------------------------#
            if len(results[0]) <= 0:
                f.close()
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        top_100     = np.argsort(top_conf)[::-1][:self.max_boxes]
        top_boxes   = top_boxes[top_100]
        top_conf    = top_conf[top_100]
        top_label   = top_label[top_100]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = float(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write(
                "%s %.10f %.10f %.10f %.10f %.10f\n"
                % (predicted_class, score, left, top, right, bottom)
            )

        f.close()
        return 
    
    def on_epoch_end(self, epoch):
        if epoch % self.period == 0 and self.eval_flag:
            if os.path.exists(self.map_out_path):
                shutil.rmtree(self.map_out_path)
            os.makedirs(self.map_out_path)
            if not os.path.exists(os.path.join(self.map_out_path, "ground-truth")):
                os.makedirs(os.path.join(self.map_out_path, "ground-truth"))
            if not os.path.exists(os.path.join(self.map_out_path, "detection-results")):
                os.makedirs(os.path.join(self.map_out_path, "detection-results"))
            module_training_states = [
                (module, module.training) for module in self.net.modules()
            ]
            self.net.eval()
            print("Get map.")
            image_sizes = {}
            for annotation_line in tqdm(self.val_lines):
                line        = annotation_line.split()
                image_id    = os.path.basename(line[0]).split('.')[0]
                #------------------------------#
                #   读取图像并转换成RGB图像
                #------------------------------#
                image       = Image.open(line[0])
                image_sizes[image_id] = {"width": image.width, "height": image.height}
                #------------------------------#
                #   获得预测框
                #------------------------------#
                image_name  = os.path.basename(line[0]).casefold()
                if image_name not in self.val_ground_truth:
                    raise KeyError(f"验证影像不在 val.json 中：{line[0]}")
                gt_boxes    = self.val_ground_truth[image_name]
                #------------------------------#
                #   获得预测txt
                #------------------------------#
                self.get_map_txt(image_id, image, self.class_names, self.map_out_path)
                
                #------------------------------#
                #   获得真实框txt
                #------------------------------#
                with open(os.path.join(self.map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                    for box in gt_boxes:
                        left, top, right, bottom, obj = box
                        obj_name = self.class_names[obj]
                        new_f.write(
                            "%s %.10f %.10f %.10f %.10f\n"
                            % (obj_name, left, top, right, bottom)
                        )
            for module, was_training in module_training_states:
                module.training = was_training
            with open(os.path.join(self.map_out_path, "image_sizes.json"), "w", encoding="utf-8") as f:
                json.dump(image_sizes, f, ensure_ascii=False)
                        
            print("Calculate Map.")
            # 论文权重选择固定使用 COCO AP50。评价失败必须直接报错，禁止静默
            # 回退到 VOC mAP 后仍将权重误标为 COCO best_map50。
            temp_map = get_coco_map(class_names = self.class_names, path = self.map_out_path)[1]
            temp_map = float(temp_map)
            map50_improved = temp_map > self.best_map50
            if map50_improved:
                self.best_map50 = temp_map
                self.best_map50_epoch = epoch
            self.maps.append(temp_map)
            self.epoches.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_map.txt"), 'a') as f:
                f.write(str(temp_map))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches, self.maps, 'red', linewidth = 2, label='train map')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Map %s'%str(self.MINOVERLAP))
            plt.title('A Map Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_map.png"))
            plt.cla()
            plt.close("all")

            print("Get map done.")
            shutil.rmtree(self.map_out_path)
            return {
                "map50": temp_map,
                "improved": map50_improved,
                "best_map50": self.best_map50,
                "best_epoch": self.best_map50_epoch,
            }
        return None

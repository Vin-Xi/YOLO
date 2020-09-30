
    import cv2
    import numpy as np
    import torch
    import os.path as ospah
    import os

    def transform_prediction(outputs, imgs, in_size):
        # get the original image dimensions
        image_dimensions = [[img.shape[0], img.shape[1]] for img in imgs]
        image_dimensions = torch.tensor(image_dimensions, dtype=torch.float)
        image_dimensions = torch.index_select(image_dimensions, 0, outputs[:, 0].long())
        in_size = torch.tensor(in_size, dtype=torch.float)
        scale_factors = torch.min(in_size / image_dimensions, 1)[0].unsqueeze(-1)
        outputs[:, [1, 3]] -= (in_size[1] - scale_factors * image_dimensions[:, 1].unsqueeze(-1)) / 2
        outputs[:, [2, 4]] -= (in_size[0] - scale_factors * image_dimensions[:, 0].unsqueeze(-1)) / 2
        outputs[:, 1:5] /= scale_factors
        outputs[:, 1:5] = torch.clamp(outputs[:, 1:5], 0)
        outputs[:, [1, 3]] = torch.min(outputs[:, [1, 3]], image_dimensions[:, 1].unsqueeze(-1))
        outputs[:, [2, 4]] = torch.min(outputs[:, [2, 4]], image_dimensions[:, 0].unsqueeze(-1))
        return outputs


    def make_pred(detection, o_thres, n_thres):
        detection = new_box(detection)
        output = torch.tensor([], dtype=torch.float)

        for n_b in range(detection.size(0)):
            bounding_boxes = detection[n_b]
            bounding_boxes = bounding_boxes[bounding_boxes[:, 4] > o_thres]

            if len(bounding_boxes) == 0:
                continue
            pred_score, pred_index = torch.max(bounding_boxes[:, 5:], 1)
            pred_score = pred_score.unsqueeze(-1)
            pred_index = pred_index.float().unsqueeze(-1)
            bounding_boxes = torch.cat((bounding_boxes[:, :5], pred_score, pred_index), dim=1)
            pred_classes = torch.unique(bounding_boxes[:, -1])
            #Do Non max suppression for each class 
            for cls in pred_classes:
                bboxes_cls = bounding_boxes[bounding_boxes[:, -1] == cls]   # select boxes that predict the class
                _, sort_indices = torch.sort(bboxes_cls[:, 4], descending=True)
                bboxes_cls = bboxes_cls[sort_indices]   # sort by objectness score
                bi = 0
                while bi + 1 < bboxes_cls.size(0):
                    ious = com_iou(bboxes_cls[bi], bboxes_cls[bi+1:])
                    bboxes_cls = torch.cat([bboxes_cls[:bi+1], bboxes_cls[bi+1:][ious < n_thres]])
                    bi += 1
                batch_idx_add = torch.full((bboxes_cls.size(0), 1), n_b)
                bboxes_cls = torch.cat((batch_idx_add, bboxes_cls), dim=1)
                output = torch.cat((output, bboxes_cls))
        return output

    def new_box(bounding_boxes):
        newbboxes = bounding_boxes.clone()
        newbboxes[:, :, 0] = bounding_boxes[:, :, 0] - bounding_boxes[:, :, 2] / 2
        newbboxes[:, :, 1] = bounding_boxes[:, :, 1] - bounding_boxes[:, :, 3] / 2
        newbboxes[:, :, 2] = bounding_boxes[:, :, 0] + bounding_boxes[:, :, 2] / 2
        newbboxes[:, :, 3] = bounding_boxes[:, :, 1] + bounding_boxes[:, :, 3] / 2
        return newbboxes

    def com_iou(box, c_boxes):
        x1,y1,x2,y2 = box[:4]
        c_x1, c_y1, c_x2, c_y2 = c_boxes[:, :4].transpose(0, 1)
        intercept_x1 = torch.max(x1, c_x1)
        intercept_y1 = torch.max(y1, c_y1)
        intercept_x2 = torch.min(x2, c_x2)
        intercept_y2 = torch.min(y2, c_y2)
        intersection= torch.clamp(intercept_x2 - intercept_x1 + 1, 0) * torch.clamp(intercept_y2 - intercept_y1 + 1, 0)
        target_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        comp_areas = (c_x2 - c_x1 + 1) * (c_y2 - c_y1 + 1)
        union = comp_areas + target_area - intersection
        ious = intersection / union
        return ious

    def input_pipeline(impath):
        if ospah.isdir(impath):
            imlist = [ospah.join(impath, img) for img in os.listdir(impath)]
        elif ospah.isfile(impath):
            imlist = [impath]
        imgs = [cv2.imread(path) for path in imlist]
        return imlist, imgs

    def cv2_tensor(img, size):
        img = resize(img, size)
        img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).float() / 255.0
        return img

    def resize(img, size):
        height, width = img.shape[0:2]
        new_height, new_width = size
        scale = min(new_height / height, new_width / width)
        img_h, img_w = int(height * scale), int(width * scale)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((new_height, new_width, 3), 128.0)
        canvas[(new_height - img_h) // 2 : (new_height - img_h) // 2 + img_h, (new_width - img_w) // 2 : (new_width-img_w) // 2 + img_w, :] = img
        return canvas


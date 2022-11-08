def get_params():
    param_dict = {} 
    param_dict['roi_batch_size'] = 32
    param_dict['lr'] = 0.001
    param_dict['momentum'] = 0.9
    param_dict['gamma_step'] = 10
    param_dict['gamma'] = 0.1
    param_dict['freeze_at'] = 1
    param_dict['rpn_bbox_loss_weight'] = 1.0
    param_dict['roi_bbox_loss_weight'] = 1.0
    param_dict['roi_score_thresh'] = 0.5
    param_dict['roi_nms_thresh'] = 0.3
    return param_dict
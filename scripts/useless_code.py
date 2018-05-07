
def yolo_imagenet200():
    all_task = []
    max_num = 500
    all_task = []
    machine_ips = []
    #for monitor_train_only in [True]:
    _report = {}
    training_time_key = 'Time(s)'
    _training_time = {training_time_key: {}}
    num_extra_convs = 3
    #suffix = '_1'
    suffix = '_2gpu'
    #suffix = '_xx'
    #dataset_ops_template = []
    #dataset_ops = []
    _num_param = {}
    num_param_key = 'Param'
    _num_param[num_param_key] = {}
    batch_size_factor = 1
    suffix = '_batchSizeFactor{}'.format(batch_size_factor) \
            if batch_size_factor != 1 else ''
    #suffix = '{}{}'.format(suffix, '_256e')
    suffix = '_withNoBB' 
    suffix = '' 
    effective_batch_size = 64 * 2
    #max_iters=1
    def gen_anchor_bias(num_anchor):
        if num_anchor == 0:
            return None
        if num_anchor == 2:
            return [4, 8, 8, 4]
        result = []
        n = int(np.sqrt(num_anchor))
        assert n * n == num_anchor
        step = 12.0 / (n + 1)
        for i in xrange(n):
            for j in xrange(n):
                result.append((i + 1) * step + 1)
                result.append((j + 1) * step + 1)
        return result
    num_anchor = 9
    anchor_bias = gen_anchor_bias(num_anchor)
    anchor_bias = None
    yolo_softmax_norm_by_valid = True
    #yolo_softmax_norm_by_valid = False
    multi_feat_anchor = [
            {   
                'feature': 'dark4c/leaky', 
                'extra_conv_channels': [512, 1024],
                'extra_conv_kernels': [3, 3],
                'loss_weight_multiplier': 1,
                'anchor_bias': [1, 1, 1, 2, 2, 1, 2, 2], 
                #'stride': 8
            },
            {
                'feature': 'dark5e/leaky', 
                'extra_conv_channels': [1024],
                'extra_conv_kernels': [3],
                #'anchor_bias': [4, 4, 2, 4, 4, 2], 
                'loss_weight_multiplier': 1,
                'anchor_bias': [2, 2, 1, 2, 2, 1], 
                #'stride': 16
            }, 
            {   
                'feature': 'dark6e/leaky', 
                'extra_conv_channels': [1024, 1024, 1024],
                'extra_conv_kernels': [3, 3, 3],
                'loss_weight_multiplier': 1,
                'anchor_bias': [2, 2, 4, 4, 4, 8, 8, 4, 8, 8],
            }
            ]
    #multi_feat_anchor = [
            #{   
                #'feature': 'dark6e/leaky', 
                #'extra_conv_channels': [1024, 1024, 1024],
                #'extra_conv_kernels': [3, 3, 3],
                #'loss_weight_multiplier': 1,
                ##'stride': 32,
                ##'anchor_bias': [6.63,11.38,9.42,5.11,16.62,10.52]
                #'anchor_bias': [1.08,1.19,3.42,4.41,6.63,11.38,9.42,5.11,16.62,10.52]
                ##'anchor_bias': [4, 4, 2, 4, 4, 2]
            #},
            #]
    multi_feat_anchor = []
    num_anchor = 5
    multibin_wh = False
    multibin_wh_low = 0;
    multibin_wh_high = 13
    all_multibin_wh_count = [16]
    #all_multibin_wh_count = [16, 32, 48, 64];
    yolo_rescore = True
    yolo_xy_kl_distance = False
    yolo_obj_only = False
    yolo_exp_linear_wh = False
    yolo_obj_nonobj_align_to_iou = False
    yolo_obj_ignore_center_around = False
    yolo_obj_kl_distance = False
    yolo_obj_set1_center_around = False
    yolo_blame = 'xy.wh.obj.cls.nonobj'
    yolo_blame = 'xy.wh.obj.nonobj.cls'
    yolo_blame = ''
    yolo_deconv_to_increase_dim = False
    yolo_deconv_to_increase_dim_adapt_bias = True
    yolo_anchor_aligned_images = 1280000000
    yolo_anchor_aligned_images = 12800
    #yolo_anchor_aligned_images = 0
    yolo_nonobj_extra_power = 0
    yolo_obj_extra_power = 0
    yolo_disable_data_augmentation = False
    yolo_disable_data_augmentation_except_shift = False
    #yolo_anchor_aligned_images = 0
    bn_no_train = False
    yolo_coords_only = False
    if yolo_coords_only:
        yolo_object_scale = 0
        yolo_noobject_scale = 0
        yolo_class_scale = 0
    else:
        yolo_object_scale = 5
        yolo_noobject_scale = 1
        yolo_class_scale = 1
    yolo_avg_replace_max = False
    yolo_per_class_obj = False
    max_iters = '128e'
    data = 'voc20'
    #data = 'brand1048'
    #data = 'office100_v1'
    if data == 'voc20':
        max_iters = 10000
    elif data == 'fridge_clean':
        max_iters = 10000
    #max_iters=10000 / batch_size_factor
    burn_in = '5e'
    burn_in = ''
    burn_in_power = 1

    yolo_full_gpu = True
    #yolo_full_gpu = False
    test_tree_cls_specific_th_by_average = 1.2
    test_tree_cls_specific_th_by_average = None
    yolo_angular_loss = True
    yolo_angular_loss = False
    no_bias = 'conf'
    no_bias = ''
    init_from = {'data': 'imagenet', 'net': 'darknet19_448', 'expid': 'A'}
    #init_from = {'data': 'office_v2.1', 'net': 'darknet19_448', 
        #'expid': 'A_burnIn5e.1_tree_initFrom.imagenet.A'}
    init_from = {}
    max_iters = '128e'
    #max_iters = 10000
    #max_iters = 11000
    num_bn_fix = 0
    kwargs_template = dict(
            detmodel='yolo',
            max_iters=max_iters,
            #max_iters='80e',
            #max_iters='256e',
            #max_iters=max_iters,
            #max_iters=10000,
            #yolo_blame=yolo_blame,
            #expid=expid,
            #yolo_jitter=0,
            #yolo_hue=0,
            #yolo_test_fix_xy = True,
            #yolo_test_fix_wh = True,
            #yolo_extract_target_prediction = True,
            #yolo_max_truth=300,
            #yolo_exposure=1,
            #test_on_train = True,
            #yolo_saturation=1,
            #yolo_random_scale_min=1,
            #yolo_random_scale_max=1,
            #expid='A_multibin_wh_0_13_16_no2-wh',
            #expid='baseline_2',
            #snapshot=1000,
            #snapshot=0,
            #target_synset_tree='./aux_data/yolo/9k.tree',
            #target_synset_tree='./data/{}/tree.txt'.format(data),
            #dataset_ops=dataset_ops,
            #effective_batch_size=1,
            #num_anchor=3,
            #num_anchor=num_anchor,
            #force_train=True,
            #force_evaluate=True,
            #debug_detect=True,
            #force_predict=True,
            #extract_features='angular_loss.softmax_loss.o_obj_loss.xy_loss.wh_loss.o_noobj_loss',
            #data_dependent_init=True,
            #restore_snapshot_iter=-1,
            #display=0,
            #region_debug_info=10,
            #display=100,
            #stagelr=stagelr,
            #anchor_bias=anchor_bias,
            #test_input_sizes=[288, 416, 480, 608],
            #test_input_sizes=[2080],
            #test_input_sizes=[416, 544, 608, 992, 1024],
            #test_input_sizes=[992],
            #test_input_sizes=[416 * 3],
            #stageiter=[1000, 1000000],
            #stageiter=[100,5000,9000,10000000],
            #stageiter = (np.asarray([5000, 9000, 1000000]) / batch_size_factor).tolist(),
            #stagelr = (np.asarray([0.001,0.0001,0.00001]) * batch_size_factor).tolist(),
            #stagelr=[0.0001,0.001,0.0001,0.0001],
            #burn_in=100,
            #class_specific_nms=False,
            #basemodel='./output/imagenet_darknet19_448_A/snapshot/model_iter_570640.caffemodel',
            #effective_batch_size=effective_batch_size,
            #solver_debug_info=True,
            yolo_test_maintain_ratio = True,
            ovthresh = [0,0.1,0.2,0.3,0.4,0.5])

    if effective_batch_size != 64:
        kwargs_template['effective_batch_size'] = effective_batch_size

    if yolo_disable_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
        kwargs_template['yolo_fix_offset'] = True
        kwargs_template['yolo_mirror'] = False
    elif yolo_disable_data_augmentation_except_shift:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1

    continue_less_data_augmentation = False
    if continue_less_data_augmentation:
        kwargs_template['yolo_jitter'] = 0
        kwargs_template['yolo_hue'] = 0
        kwargs_template['yolo_exposure'] = 1
        kwargs_template['yolo_saturation'] = 1
        kwargs_template['yolo_random_scale_min'] = 1
        kwargs_template['yolo_random_scale_max'] = 1
    #import ipdb;ipdb.set_trace()
    #if batch_size_factor == 2:
        #burn_in = '5e'
        #burn_in_power = 1

    #adv = [(False, True), (True, False), (True, True)]
    #max_iters = 4
    #for multibin_wh_count in all_multibin_wh_count:
    multibin_wh_count = all_multibin_wh_count[0]
    #for yolo_obj_ignore_center_around, yolo_obj_kl_distance in adv:
    yolo_fixed_target = False
    yolo_obj_nonobj_nopenaltyifsmallthaniou = False
    yolo_obj_cap_center_around = False
    yolo_multibin_xy = False
    yolo_sigmoid_xy = True
    yolo_delta_region3 = False
    yolo_background_class = False
    #yolo_background_class = False
    yolo_use_background_class_to_reduce_obj = 0.4
    #for multibin_xy_count in [32, 16, 8]:
    multibin_xy_count = 4
    #res_loss = True
    res_loss = False
    yolo_use_background_class_to_reduce_obj = 1
    monitor_train_only = False
    #monitor_train_only = True
    #for yolo_use_background_class_to_reduce_obj in [1, 0.8, 0.6, 0.4, 0.2]:
    dataset = TSVDataset(data)
    yolo_low_shot_regularizer = False
    #full_labels = dataset.load_labelmap()
    full_labels = []
    #for low_shot_label in full_labels:
    yolo_random_scale_max = 2
    scale_relative_input = True
    scale_relative_input = False
    nms_type = 'LinearSoft'
    nms_type = 'GaussianSoft'
    nms_type = ''
    gaussian_nms_sigma = 0.5
    scale_constrained_by_one_box_area_min = 0.001
    last_fixed_param = 'dark5e/leaky'
    last_fixed_param = ''
    #for low_shot_label in ['']:
    #for low_shot_label in [full_labels[0]]:
    full_labels.insert(0, '')
    #for low_shot_label in full_labels[:5]:
    low_shot_label = ''
    residual_loss = True
    residual_loss = False
    residual_loss_froms = ['extra_conv19/leaky', 'extra_conv20/leaky']
    #for low_shot_label in ['']:
    #for extra_conv_kernel in [[1, 1], [1, 3], [3, 1]]:
    yolo_disable_no_penalize_if_iou_large = False
    cutout_prob = -1
    expid_prefix = 'A'
    #expid_prefix = 'debug3'
    yolo_random_scale_min = 0.25
    yolo_random_scale_max = 2
    #yolo_random_scale_min = 2
    #yolo_random_scale_max = 4
    #net_input_size_min = 416 * 2
    #net_input_size_max = 416 * 2
    net_input_size_min = 416
    net_input_size_max = 416
    yolo_softmax_extra_weight = 0.2
    ignore_negative_first_batch = False
    yolo_not_ignore_negative_seen_images = 0
    yolo_xywh_norm_by_weight_sum = False
    #yolo_softmax_extra_weight = 1
    for extra_conv_kernel in [[3, 3, 3]]:
    #for extra_conv_channels in [[1024, 512, 512], [512, 512, 1024], [512, 1024,
        #512]]:
        extra_conv_channels = [1024, 1024, 1024]
    #for extra_conv_channels in [[1024, 1024]]:
    #for extra_conv_channels in [[1024, 1024, 1024]]:
    #for extra_conv_kernel in [[3,3,3], [1, 1, 1], [1, 3, 1], [3, 1, 1], [1, 1, 3], [3,
        #3, 1], [3, 1, 3], [1, 3, 3]]:
    #for extra_conv_kernel in [[1, 3, 1]]:
        if len(low_shot_label) > 0:
            dataset_ops = [{'op':'low_shot', 'labels': 'dog', 'num_train': 1}]
            dataset_ops[0]['labels'] = low_shot_label
            dataset_ops[0]['labels_idx'] = full_labels.index(low_shot_label)
        else:
            dataset_ops = [{'op':'remove'},
                    {'op':'add',
                     'name':'office_v2.11_with_bb',
                     'source':'train',
                     'weight': 3},
                    #{'op': 'add',
                     #'name': 'office_v2.1_no_bb',
                     #'source': 'train',
                     #'weight': 1},
                    ]
            dataset_ops = []
            #for multibin_xy_count in [4]:
        yolo_multibin_xy_low = 0.5 / multibin_xy_count
        #multibin_xy_count = 16
        yolo_multibin_xy_high = 1 - 0.5 / multibin_xy_count
        yolo_multibin_xy_count = multibin_xy_count
        #for yolo_obj_kl_distance in [False]:
        yolo_obj_kl_distance = False
    #for yolo_object_scale in [40, 60]:
        #for monitor_train_only in [False, True]:
        #for monitor_train_only in [True]:
        #for yolo_iou_th_to_use_bkg_cls in [0.1]:
        yolo_iou_th_to_use_bkg_cls = 0.1
        #for last_conv_bias in [True]:
        last_conv_bias = True
        #for rotate_max in [5, 10, 15, 20]:
        rotate_max = 10
        rotate_max = 0
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, True), (True, False)]:
        #for incorporate_at_least_one_box, scale_constrained_by_one_box_area in [(False, False)]:
        incorporate_at_least_one_box, scale_constrained_by_one_box_area = False, False
        #for yolo_angular_loss_weight in [0.1, 1, 10]:
        #for yolo_angular_loss_weight in [1]:
        yolo_angular_loss_weight = 1
        #for data in ['fridge_clean', 'voc20', 'CARPK']:
        #for data in ['office_v2.1']:
        #for data in ['CARPK_select.5.5.nokeep']:
        for data in ['imagenet200']:
        #for data in ['coco_phone']:
        #for data in ['coco2017']:
            if data.startswith('office'):
                yolo_tree = True
            else:
                yolo_tree = False
            if test_tree_cls_specific_th_by_average:
                assert yolo_tree
        #rotate_max = 10
        #rotate_max = 0
            rotate_with_90 = False
        #for yolo_random_scale_min in [0.5, 0.75]:
        #for monitor_train_only in [False]:
        #monitor_train_only = True
        #monitor_train_only = True
    #for monitor_train_only in [True]:
    #for monitor_train_only in [False, True]:
            #for add_reorg, num_extra_convs in [(True, [6]), (True, [7])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(True, [3, 3, 3, 3])]:
            #for add_reorg, num_extra_convs in [(False, [3]), (True, [3])]:
            #for add_reorg, num_extra_convs in [(True, [3])]:
            for add_reorg, num_extra_convs in [(True, [3])]:
        #for add_reorg, num_extra_convs in all_extra_option:
        #for weight_decay in [0.0001, 0.00005]:
        #for weight_decay in [0.0005]:
            #weight_decay = 0.0005
            #weight_decay = 0
            #for net in ['darknet19', 'resnet18', 'resnet34', 'resnet50', 'resnet101']:
            #for net in ['resnet34']:
            #net = 'darknet19'
            #for net in ['resnet101']:
            #for yolo_coord_scale in [1]:
                yolo_coord_scale = 1
                #for data in ['voc20', 'voc2012', 'coco2017', 'imagenet']:
                #for data in ['coco2017']:
                #for weight_decay in [0.001, 0.005, 0.01, 0.05]:
                weight_decay = 0.0005
                #for net in ['resnet34', 'darknet19_448']:
                #for net in ['darknet19_448', 'resnet34']:
                #for net in ['resnet34']:
                for net in ['resnet34']:
                #for net in ['darknet19_448', 'resnet34', 'resnet101']:
                    if len(burn_in) > 0 and data != 'voc20':
                        kwargs_template['stageiter'] = ['60e', '90e', '900e'] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()
                    elif len(burn_in) > 0:
                        kwargs_template['stageiter'] = [5000, 9000, 100000] 
                        kwargs_template['stagelr'] = (np.asarray([0.001,0.0001,0.00001]) *
                                batch_size_factor).tolist()

                    kwargs = copy.deepcopy(kwargs_template)
                    expid = expid_prefix
                    expid = expid + ('_noreorg' if not add_reorg else '')
                    if len(num_extra_convs) == 0 and num_extra_convs == 0:
                        expid = expid + '_noextraconv'
                        kwargs['num_extra_convs'] = num_extra_convs
                    elif len(num_extra_convs) == 1 and num_extra_convs[0] != 3 \
                            or len(num_extra_convs) > 1:
                        expid = expid + '_extraconv{}'.format(
                                '_'.join(map(str, num_extra_convs)))
                        kwargs['num_extra_convs'] = num_extra_convs
                    if multibin_wh:
                        expid = expid + '_multibin_wh_{}_{}_{}'.format(multibin_wh_low,
                                multibin_wh_high, multibin_wh_count)
                        kwargs['multibin_wh'] = multibin_wh
                        kwargs['multibin_wh_low'] = multibin_wh_low
                        kwargs['multibin_wh_high'] = multibin_wh_high
                        kwargs['multibin_wh_count'] = multibin_wh_count
                    if not add_reorg:
                        kwargs['add_reorg'] = add_reorg
                    if num_anchor != 5:
                        expid = '{}_numAnchor{}'.format(expid, num_anchor)
                        assert len(anchor_bias) == 2 * num_anchor
                        kwargs['anchor_bias'] = anchor_bias
                    if not yolo_rescore:
                        expid = '{}_{}'.format(expid, 'norescore')
                        kwargs['yolo_rescore'] = yolo_rescore
                    if yolo_obj_ignore_center_around:
                        expid = '{}_{}'.format(expid, 'ignore')
                        kwargs['yolo_obj_ignore_center_around'] = yolo_obj_ignore_center_around
                    if yolo_obj_kl_distance:
                        expid = '{}_{}'.format(expid, 'objkl')
                        kwargs['yolo_obj_kl_distance'] = yolo_obj_kl_distance
                    if yolo_xy_kl_distance:
                        expid = '{}_xykl'.format(expid)
                        kwargs['yolo_xy_kl_distance'] = yolo_xy_kl_distance
                    if yolo_obj_only:
                        expid = '{}_objonly'.format(expid)
                        kwargs['yolo_obj_only'] = yolo_obj_only
                    if yolo_exp_linear_wh:
                        expid = '{}_explinearwh'.format(expid)
                        kwargs['yolo_exp_linear_wh'] = yolo_exp_linear_wh
                    if yolo_obj_nonobj_align_to_iou:
                        expid = '{}_nonobjtoiou'.format(expid)
                        kwargs['yolo_obj_nonobj_align_to_iou'] = yolo_obj_nonobj_align_to_iou
                    if yolo_obj_set1_center_around:
                        expid = '{}_around1'.format(expid)
                        kwargs['yolo_obj_set1_center_around'] = yolo_obj_set1_center_around
                    if yolo_obj_nonobj_nopenaltyifsmallthaniou:
                        expid = '{}_nolosssmallthaniou'.format(expid)
                        kwargs['yolo_obj_nonobj_nopenaltyifsmallthaniou'] = yolo_obj_nonobj_nopenaltyifsmallthaniou
                    if yolo_obj_cap_center_around:
                        expid = '{}_objcaparound'.format(expid)
                        kwargs['yolo_obj_cap_center_around'] = yolo_obj_cap_center_around
                    if weight_decay != 0.0005:
                        expid = '{}_decay{}'.format(expid, weight_decay)
                        kwargs['weight_decay'] = weight_decay
                    if yolo_fixed_target:
                        expid = '{}_fixedtarget'.format(expid)
                        kwargs['yolo_fixed_target'] = yolo_fixed_target
                    if yolo_deconv_to_increase_dim:
                        expid = '{}_deconvincreasedim'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim'] = True
                    if yolo_coord_scale != 1:
                        expid = '{}_coordscale{}'.format(expid,
                                yolo_coord_scale)
                        kwargs['yolo_coord_scale'] = yolo_coord_scale
                    if not yolo_deconv_to_increase_dim_adapt_bias:
                        expid = '{}_nobiaseadapt'.format(expid)
                        kwargs['yolo_deconv_to_increase_dim_adapt_bias'] = \
                            False
                    if yolo_anchor_aligned_images != 12800:
                        expid = '{}_align{}'.format(expid,
                                yolo_anchor_aligned_images)
                        kwargs['yolo_anchor_aligned_images'] = yolo_anchor_aligned_images
                    if yolo_nonobj_extra_power != 0:
                        expid = '{}_nonobjpower{}'.format(expid,
                                yolo_nonobj_extra_power)
                        kwargs['yolo_nonobj_extra_power'] = yolo_nonobj_extra_power
                    if yolo_obj_extra_power != 0:
                        expid = '{}_objpower{}'.format(expid,
                                yolo_obj_extra_power)
                        kwargs['yolo_obj_extra_power'] = yolo_obj_extra_power
                    if yolo_multibin_xy:
                        kwargs['yolo_multibin_xy'] = yolo_multibin_xy
                        kwargs['yolo_multibin_xy_low'] = yolo_multibin_xy_low
                        kwargs['yolo_multibin_xy_high'] = yolo_multibin_xy_high
                        kwargs['yolo_multibin_xy_count'] = yolo_multibin_xy_count
                        expid = '{}_multibinXY{}'.format(expid,
                                yolo_multibin_xy_count)
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'select_top':
                        expid = '{}_selectTop{}'.format(expid,
                                dataset_ops[0]['num_top'])
                    if len(dataset_ops) == 1 and \
                            dataset_ops[0]['op'] == 'low_shot':
                        low_shot_labels = dataset_ops[0]['labels']
                        low_shot_num_train = dataset_ops[0]['num_train']
                        expid = '{}_lowShot.{}.{}'.format(expid, 
                                low_shot_labels,
                                low_shot_num_train)
                    if len(dataset_ops) > 0:
                        kwargs['dataset_ops'] = dataset_ops
                        expid = '{}_dataop{}'.format(expid,
                                hash(json.dumps(dataset_ops))%10000)
                    if yolo_disable_data_augmentation:
                        expid = '{}_noAugmentation'.format(expid)
                    if yolo_disable_data_augmentation_except_shift:
                        expid = '{}_noAugExpShift'.format(expid)
                    if bn_no_train:
                        expid = '{}_bnNoTrain'.format(expid)
                        kwargs['bn_no_train'] = bn_no_train
                    if yolo_object_scale != 5:
                        expid = '{}_objscale{}'.format(expid, yolo_object_scale)
                        kwargs['yolo_object_scale'] = yolo_object_scale
                    if yolo_noobject_scale != 1:
                        expid = '{}_noobjScale{}'.format(expid,
                                yolo_noobject_scale)
                        kwargs['yolo_noobject_scale'] = yolo_noobject_scale
                    if yolo_class_scale != 1:
                        expid = '{}_clsScale{}'.format(expid,
                                yolo_class_scale)
                        kwargs['yolo_class_scale'] = yolo_class_scale
                    if yolo_avg_replace_max:
                        expid = '{}_avgReplaceMax'.format(expid)
                        kwargs['yolo_avg_replace_max'] = yolo_avg_replace_max
                    if not yolo_sigmoid_xy:
                        expid = '{}_nosigmoidXY'.format(expid)
                        kwargs['yolo_sigmoid_xy'] = yolo_sigmoid_xy
                    if yolo_delta_region3:
                        expid = '{}_deltaRegion3'.format(expid)
                        kwargs['yolo_delta_region3'] = yolo_delta_region3
                    if yolo_background_class:
                        expid = '{}_bkgCls{}'.format(expid,
                                yolo_use_background_class_to_reduce_obj)
                        kwargs['yolo_background_class'] = True
                        kwargs['yolo_use_background_class_to_reduce_obj'] = yolo_use_background_class_to_reduce_obj
                        if yolo_iou_th_to_use_bkg_cls != 1:
                            expid = '{}_iouTh{}'.format(expid,
                                    yolo_iou_th_to_use_bkg_cls)
                            kwargs['yolo_iou_th_to_use_bkg_cls'] = yolo_iou_th_to_use_bkg_cls
                    if res_loss:
                        expid = '{}_resLoss'.format(expid)
                        kwargs['res_loss'] = res_loss
                        kwargs['skip_genprototxt'] = True
                    if yolo_per_class_obj:
                        expid = '{}_perClassObj'.format(expid)
                        kwargs['yolo_per_class_obj'] = yolo_per_class_obj
                    if not last_conv_bias:
                        expid = '{}_noBiasLastConv'.format(expid)
                        kwargs['yolo_last_conv_bias'] = last_conv_bias
                    if yolo_low_shot_regularizer:
                        expid = '{}_lowShotEqualNorm'.format(expid)
                        kwargs['yolo_low_shot_regularizer'] = True
                    if yolo_full_gpu:
                        expid = '{}_fullGpu'.format(expid)
                        kwargs['yolo_full_gpu'] = yolo_full_gpu
                    if burn_in != '':
                        expid = '{}_burnIn{}.{}'.format(expid, burn_in,
                                burn_in_power)
                        kwargs['burn_in'] = burn_in
                        kwargs['burn_in_power'] = burn_in_power
                    if rotate_max != 0:
                        kwargs['rotate_max'] = rotate_max
                        expid = '{}_rotate{}'.format(expid, rotate_max)
                    if rotate_with_90:
                        expid = '{}_with90'.format(expid)
                        kwargs['rotate_with_90'] = True
                    if yolo_random_scale_min != 0.25:
                        expid = '{}_randomScaleMin{}'.format(expid, 
                                yolo_random_scale_min)
                        kwargs['yolo_random_scale_min'] = yolo_random_scale_min
                    if yolo_random_scale_max != 2:
                        expid = '{}_randomScaleMax{}'.format(expid,
                                yolo_random_scale_max)
                        kwargs['yolo_random_scale_max'] = yolo_random_scale_max
                    if scale_relative_input:
                        expid = '{}_RelativeScale2'.format(expid)
                        kwargs['scale_relative_input'] = scale_relative_input
                    if nms_type != 'Standard' and nms_type != '':
                        if nms_type == 'LinearSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.LinearSoft 
                        if nms_type == 'GaussianSoft':
                            kwargs['nms_type'] = caffe.proto.caffe_pb2.RegionPredictionParameter.GaussianSoft
                        if gaussian_nms_sigma != 0.5:
                            kwargs['gaussian_nms_sigma'] = 0.5
                    if incorporate_at_least_one_box:
                        expid = '{}_atLeastOneBB'.format(expid)
                        kwargs['incorporate_at_least_one_box'] = incorporate_at_least_one_box
                    if scale_constrained_by_one_box_area:
                        expid = '{}_scaleConstrainedByOne'.format(expid)
                        kwargs['scale_constrained_by_one_box_area'] = scale_constrained_by_one_box_area
                        if scale_constrained_by_one_box_area_min != 0.001:
                            expid = '{}_Min{}'.format(expid, scale_constrained_by_one_box_area_min)
                            kwargs['scale_constrained_by_one_box_area_min'] = scale_constrained_by_one_box_area_min
                    if yolo_tree:
                        expid = '{}_tree'.format(expid)
                        kwargs['yolo_tree'] = yolo_tree
                        if test_tree_cls_specific_th_by_average is not None:
                            kwargs['test_tree_cls_specific_th_by_average'] = test_tree_cls_specific_th_by_average
                    if len(init_from) > 0:
                        assert net == init_from['net']
                        if len(init_from['expid']) > 5:
                            expid = '{}_initFrom'.format(expid)
                        else:
                            expid = '{}_initFrom.{}.{}'.format(expid,
                                    init_from['data'], init_from['expid'])
                        c = CaffeWrapper(data=init_from['data'], 
                                net=init_from['net'],
                                expid=init_from['expid'])
                        kwargs['basemodel'] = c.best_model().model_param
                    if yolo_angular_loss:
                        expid = '{}_AngularRegulizer'.format(expid)
                        kwargs['yolo_angular_loss'] = True
                        if yolo_angular_loss_weight != 1:
                            expid = '{}Weight{}'.format(expid,
                                    yolo_angular_loss_weight)
                            kwargs['yolo_angular_loss_weight'] = yolo_angular_loss_weight
                    if len(no_bias) > 0:
                        expid = '{}_noBias{}'.format(expid, no_bias)
                        kwargs['no_bias'] = no_bias
                    if net_input_size_min != 416:
                        expid = '{}_InMin{}'.format(expid, net_input_size_min)
                        kwargs['net_input_size_min'] = net_input_size_min
                    if net_input_size_max != 416:
                        expid = '{}_InMax{}'.format(expid, net_input_size_max)
                        kwargs['net_input_size_max'] = net_input_size_max
                    if any(k != 3 for k in extra_conv_kernel):
                        expid = '{}_extraConvKernel.{}'.format(expid, '.'.join(
                            map(str, extra_conv_kernel)))
                        kwargs['extra_conv_kernel'] = extra_conv_kernel
                    if any(c != 1024 for c in extra_conv_channels):
                        expid = '{}_extraChannels.{}'.format(expid,
                                '.'.join(map(str, extra_conv_channels)))
                        kwargs['extra_conv_channels'] = extra_conv_channels
                    if len(last_fixed_param) > 0:
                        expid = '{}_FixParam.{}'.format(expid,
                                last_fixed_param.replace('/', '.'))
                        kwargs['last_fixed_param'] = last_fixed_param
                    if data.startswith('office_v2.1'):
                       assert 'taxonomy_folder' not in kwargs
                       kwargs['taxonomy_folder'] = \
                            './aux_data/taxonomy10k/office/{}'.format(data)
                    if residual_loss:
                        expid = '{}_resLoss{}'.format(expid, 
                                '.'.join(map(lambda x: x.replace('/', '_'),
                                    residual_loss_froms)))
                        kwargs['residual_loss'] = residual_loss
                        kwargs['residual_loss_froms'] = residual_loss_froms
                    if yolo_disable_no_penalize_if_iou_large:
                        kwargs['yolo_disable_no_penalize_if_iou_large'] = True
                        expid = '{}_disableNoPenIfIouLarge'.format(expid)
                    if cutout_prob > 0:
                        kwargs['cutout_prob'] = cutout_prob
                        expid = '{}_cutoutProb{}'.format(expid, cutout_prob)
                    if len(multi_feat_anchor) > 0:
                        kwargs['multi_feat_anchor'] = multi_feat_anchor
                        expid = '{}_multiFeatAnchor{}.{}.{}'.format(expid,
                                len(multi_feat_anchor),
                                '.'.join(map(str, [m['loss_weight_multiplier'] for m in
                                    multi_feat_anchor])),
                                hash(json.dumps(multi_feat_anchor)))
                    if max_iters != '128e' and max_iters != 10000:
                        expid = '{}_maxIter.{}'.format(expid, max_iters)
                    if yolo_softmax_norm_by_valid:
                        expid = '{}_softmaxByValid'.format(expid)
                        kwargs['yolo_softmax_norm_by_valid'] = yolo_softmax_norm_by_valid
                    #if yolo_softmax_extra_weight != 1:
                        #assert False, 'no longer supported here'
                        #expid = '{}_softmaxWeight{}'.format(expid,
                                #yolo_softmax_extra_weight)
                        #kwargs['yolo_softmax_extra_weight'] = yolo_softmax_extra_weight
                    if num_bn_fix > 0:
                        expid = '{}_BNFix{}'.format(expid, num_bn_fix)
                        kwargs['num_bn_fix'] = num_bn_fix
                    if ignore_negative_first_batch:
                        expid = '{}_ignoreNegativeFirst'.format(expid)
                        kwargs['ignore_negative_first_batch'] = ignore_negative_first_batch
                        if yolo_not_ignore_negative_seen_images > 0:
                            expid = '{}_notIgnore{}'.format(expid,
                                    yolo_not_ignore_negative_seen_images)
                            kwargs['yolo_not_ignore_negative_seen_images'] = yolo_not_ignore_negative_seen_images
                    if yolo_xywh_norm_by_weight_sum:
                        expid = '{}_xywhNormWeight'.format(expid)
                        kwargs['yolo_xywh_norm_by_weight_sum'] = True
                    expid = expid + suffix
                    kwargs['monitor_train_only'] = monitor_train_only
                    kwargs['expid'] = expid
                    kwargs['net'] = net
                    kwargs['data'] = data
                    all_task.append(kwargs)

    #all_gpus = [[4,5,6,7], [0,1,2,3]]
    all_gpus = [[0,1,2,3], [4,5,6,7]]
    #all_gpus = [[4,5,6,7]]
    #all_gpus = [[0,1]]
    #all_gpus = [[0]]
    #all_gpus = [[0,1,2,3]]
    #all_gpus = [[4, 5, 6, 7]]
    #all_gpus = [[0,1,2,3,4,5,6,7]]
    #all_gpus = [[0]]
    machines = get_machine()
    vigs = machines['vigs']
    clusters8 = machines['clusters8']
    clusters4 = machines['clusters4']
    clusters2 = machines['clusters2']

    all_resource = []
    #for c in vigs:
        #for g in [[0,1,2,3]]:
            #all_resource += [(c, g)]
    for c in clusters8:
        #for g in [[0,1,2,3], [4,5,6,7]]:
        for g in [[0,1,2,3,4,5,6,7]]:
            all_resource += [(c, g)]
    #for c in clusters4:
        #for g in [[0,1,2,3]]:
            #all_resource += [(c, g)]
    #for c in clusters2:
        #for g in [[0,1]]:
            #all_resource += [(c, g)]
    logging.info(pformat(all_resource))
    logging.info(pformat(all_task))
    logging.info('#resource: {}'.format(len(all_resource)))
    logging.info('#task: {}'.format(len(all_task)))
    debug = True
    #all_task[0]['force_predict'] = True
    debug = False
    #return
    #tsv_file = './data/office_v2.1_with_bb/test.tsv'
    #all_task[0]['force_predict'] = True
    #task = all_task[0]
    #task['expid'] = '{}_bb_nobb'.format(task['expid'])
    ##task['expid'] = '{}_bb_only'.format(task['expid'])
    #task['class_specific_nms'] = False
    #task['yolo_test_thresh'] = 0.5
    #c = CaffeWrapper(**task)
    #c.demo(None)
    #rows = tsv_reader(tsv_file)
    #for row in rows:
        #continue
    #import ipdb;ipdb.set_trace()
    #c.demo('./data/office100_v1_with_bb/train.tsv')
    #c.demo(tsv_file)
    #c.demo('/raid/jianfw/data/office100_crawl/TermList.instagram.pinterest.scrapping.image.tsv')
    #c.demo('/raid/jianfw/work/yuxiao_crop/ring/')
    #c.demo('tmp.png')
    #all_task[0]['force_predict'] = True
    #return
    def batch_run():
        b = BatchProcess(all_resource, all_task, task_processor)
        #b._availability_check = False
        b.run()
        #if not monitor_train_only:
            #for t in all_task:
                #t['monitor_train_only'] = True
            #for i, r in enumerate(all_resource):
                #all_resource[i] = (r[0], [-1] * 4)
            #b = BatchProcess(all_resource, all_task, task_processor)
            ##b._availability_check = False
            #b.run()
    if debug:
        idx = -1
        task = all_task[idx]
        task['effective_batch_size'] = 8
        #task['use_pretrained'] = False
        #task['max_iters'] = 1
        #task['expid'] = '{}_debug'.format(expid)
        #all_task[idx]['datas'] = ['voc20', 'crawl_office_v1']
        #task['force_train'] = True
        task['debug_train'] = True
        #all_task[idx]['debug_detect'] = True
        #all_task[idx]['force_predict'] = True
        #task_processor(({}, [0]), all_task[idx])
        #task['force_evaluate'] = True
        task_processor(({}, [0]), task)
        #task_processor(all_resource[-1], task)
        #task_processor((vig[1], [0]), task)
        #import ipdb;ipdb.set_trace()
    else:
        batch_run()

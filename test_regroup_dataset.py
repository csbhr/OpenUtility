from utils.video_regroup_utils import *

# ori_root = '/home/csbhr/workspace/python/python_data/GOPRO/train'
# dest_root = '/home/csbhr/workspace/python/python_data/GOPRO/train-TypeVideo'
# VideoType2TypeVideo(ori_root, dest_root, ori_type='blur', dest_type='blur')
# VideoType2TypeVideo(ori_root, dest_root, ori_type='sharp', dest_type='gt')

# ori_root = '/home/csbhr/workspace/python/python_work/STFAN/datasets/qualitative_datasets/RealBlur/mid'
# dest_root = '/home/csbhr/workspace/python/python_work/STFAN/datasets/qualitative_datasets/RealBlur/test'
# TypeVideo2VideoType(ori_root, dest_root, ori_type='blur', dest_type='input')
# TypeVideo2VideoType(ori_root, dest_root, ori_type='gt', dest_type='GT')

# root = '/home/csbhr/workspace/deblur_result/temp_test/experiment/testSRN_Video_Deblur/result/RealBlur'
# remove_frame_prefix(root, prefix='L_')

# root = '/home/csbhr/workspace/deblur_result/dvd_results_compare/2017_CVPR_Gong/quantitative'
# remove_frame_postfix(root, postfix='_result')

# root = '/home/csbhr/workspace/temp/300epoch/real_blur/STFAN'
# add_frame_postfix(root, postfix='stfan')

# # DVD start_list=[0]
# # GOPRO start_list=[1, 4001, 3011, 1, 101, 1, 1, 1, 1, 1, 201]
# # REDS4 start_list=[0]
# # RealBlur start_list=[1, 1, 51, 1, 1, 1]
# root = '/home/csbhr/workspace/sr_result/Real/DUF'
# resort_frame_index(root, template='{:0>4}_duf', start_list=[1, 1, 2, 1, 1, 1, 1, 1, 1, 1,
#                                                             1, 1, 101, 1, 11, 1, 1, 1, 25, 13,
#                                                             1, 1, 2, 3, 1, 6, 1, 1])

# root = '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours/GOPRO/450epoch/pwc_srn_baseline'
# remove_pre_tail_frames(root, num=1)

# ori_root = '/media/csbhr/KINGSTON/epoch7_Vid4/before'
# dest_root = '/media/csbhr/KINGSTON/epoch7_Vid4/after'
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='deblur', new_postfix='srn_baseline')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='deblur', new_postfix='pwc_srn_baseline')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='deblur_iter2', new_postfix='iterate')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='deblur_iter2', new_postfix='iterate_mask')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='mask1', new_postfix='mask_stage1_1')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='mask2', new_postfix='mask_stage1_2')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='mask3', new_postfix='mask_stage1_3')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='mask2_recons', new_postfix='mask_stage2_2')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow01', new_postfix='flow_stage1_01', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow21', new_postfix='flow_stage1_21', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow12', new_postfix='flow_stage1_12', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow32', new_postfix='flow_stage1_32', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow23', new_postfix='flow_stage1_23', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow43', new_postfix='flow_stage1_43', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow12_recons', new_postfix='flow_stage2_12', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='flow32_recons', new_postfix='flow_stage2_32', ext='pt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='gt')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='edvr')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='stfan')
# # extra_frames_from_videos(ori_root, dest_root, ori_postfix='iterate_mask')


# root_list = [
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/GOPRO/iterate/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/GOPRO/iterate/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/GOPRO/iterate_mask/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/GOPRO/iterate_mask/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/RealBlur/pwc_srn_baseline/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/RealBlur/pwc_srn_baseline/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/RealBlur/iterate/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/RealBlur/iterate/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/RealBlur/iterate_mask/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/RealBlur/iterate_mask/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/REDS4/pwc_srn_baseline/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/REDS4/pwc_srn_baseline/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/REDS4/iterate/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/REDS4/iterate/flow'
#     },
#     {
#         'ori': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/intermediate_result/REDS4/iterate_mask/flow',
#         'dest': '/media/csbhr/Disk2T/BHR/Video_Deblur_Results/ours_intermediate_result/REDS4/iterate_mask/flow'
#     }
# ]
# for root in root_list:
#     ori_root = root['ori']
#     dest_root = root['dest']
#     print(">>>>  new transform >>>>")
#     print(">>>>  ori_root: {}".format(ori_root))
#     print(">>>>  dest_root: {}".format(dest_root))
#     save_flow_pt2mat(ori_root, dest_root)

window.RGBD_AUGMENTATION_REVIEW = {
  "version": 1,
  "seed": 20260820,
  "sampling": {
    "datasets": 5,
    "camera_views": 15,
    "transforms": 10,
    "review_samples": 150,
    "max_num_transforms": 1
  },
  "source": {
    "rgb": "offline three-view RGB videos",
    "depth": "LingBot v0.5 + sensor fusion visualization",
    "frame_selection": "human-approved interaction representative frames"
  },
  "versions": [
    {
      "id": "rgb_only",
      "label": "版本一：仅 RGB 增强",
      "depth_policy": "所有变换均保持深度原值"
    },
    {
      "id": "rgbd_paired",
      "label": "版本二：RGB-D 物理一致增强",
      "depth_policy": "空间遮挡同步写入无效深度；光度变换保持 metric depth"
    }
  ],
  "transforms": [
    {
      "id": "notransform",
      "label": "不增强",
      "type": "Identity",
      "family": "identity",
      "weight": 3.0,
      "probability": 0.25,
      "rgb_rule": "保持原图",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "brightness",
      "label": "亮度",
      "type": "ColorJitter",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "contrast",
      "label": "对比度",
      "type": "ColorJitter",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "saturation",
      "label": "饱和度",
      "type": "ColorJitter",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "hue",
      "label": "色相",
      "type": "ColorJitter",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "sharpness",
      "label": "锐度",
      "type": "SharpnessJitter",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "random_mask",
      "label": "随机遮挡",
      "type": "RandomMask",
      "family": "spatial_occlusion",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "同位置写入无效深度 0"
    },
    {
      "id": "random_border_cutout",
      "label": "边缘裁除",
      "type": "RandomBorderCutout",
      "family": "spatial_occlusion",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "同位置写入无效深度 0"
    },
    {
      "id": "gaussian_noise",
      "label": "高斯噪声",
      "type": "GaussianNoise",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    },
    {
      "id": "gamma_correction",
      "label": "Gamma 校正",
      "type": "GammaCorrection",
      "family": "photometric",
      "weight": 1.0,
      "probability": 0.08333333333333333,
      "rgb_rule": "应用增强",
      "rgb_only_depth_rule": "深度保持原值",
      "paired_depth_rule": "保持 metric depth 原值"
    }
  ],
  "datasets": [
    {
      "id": "leju_claw",
      "label": "夹爪采集",
      "sequence": "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003",
      "frame_index": 90,
      "original_rgb": "assets/leju_claw/original_rgb.jpg",
      "original_depth": "assets/leju_claw/original_depth.jpg",
      "depth_method": "lingbot_v05_sensor_fused",
      "samples": {
        "notransform": {
          "augmented_rgb": "assets/leju_claw/notransform/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/notransform/paired_depth.jpg",
          "rgb_changed_fraction": 0.0,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "brightness": {
          "augmented_rgb": "assets/leju_claw/brightness/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/brightness/paired_depth.jpg",
          "rgb_changed_fraction": 0.9998611111111111,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            1,
            2
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.5205995789650881
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.479734791620706
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9995833333333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.6261904723526114
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "contrast": {
          "augmented_rgb": "assets/leju_claw/contrast/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/contrast/paired_depth.jpg",
          "rgb_changed_fraction": 0.9998480902777778,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.8253163445603899
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9997526041666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.5847300017528263
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999739583333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.4984218019644142
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998177083333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "saturation": {
          "augmented_rgb": "assets/leju_claw/saturation/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/saturation/paired_depth.jpg",
          "rgb_changed_fraction": 0.9932682291666667,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.6020399728412507
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9951432291666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.7148800215365253
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9881510416666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 1.4424725624732342
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9965104166666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "hue": {
          "augmented_rgb": "assets/leju_claw/hue/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/hue/paired_depth.jpg",
          "rgb_changed_fraction": 0.9915755208333333,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.04327414289286592
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9936067708333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.04176950892026128
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9877734375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.028346659803080643
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9933463541666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "sharpness": {
          "augmented_rgb": "assets/leju_claw/sharpness/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/sharpness/paired_depth.jpg",
          "rgb_changed_fraction": 0.4017144097222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7637521733548299
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.4242447916666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7122305760615653
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.36529947916666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7457185988684061
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.4155989583333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "random_mask": {
          "augmented_rgb": "assets/leju_claw/random_mask/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/random_mask/paired_depth.jpg",
          "rgb_changed_fraction": 0.01,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.01,
          "intended_spatial_mask_fraction": 0.01,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 195,
                "left": 77,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_l",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 206,
                "left": 77,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_r",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 57,
                "left": 247,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            }
          ]
        },
        "random_border_cutout": {
          "augmented_rgb": "assets/leju_claw/random_border_cutout/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/random_border_cutout/paired_depth.jpg",
          "rgb_changed_fraction": 0.15,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.15,
          "intended_spatial_mask_fraction": 0.15,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_l",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "left",
                "pixels": 48
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_r",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "left",
                "pixels": 48
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            }
          ]
        },
        "gaussian_noise": {
          "augmented_rgb": "assets/leju_claw/gaussian_noise/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/gaussian_noise/paired_depth.jpg",
          "rgb_changed_fraction": 0.9998958333333333,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 801142325
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998567708333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 3336970300
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998958333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 1491893991
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999348958333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "gamma_correction": {
          "augmented_rgb": "assets/leju_claw/gamma_correction/augmented_rgb.jpg",
          "paired_depth": "assets/leju_claw/gamma_correction/paired_depth.jpg",
          "rgb_changed_fraction": 0.9995008680555556,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            7,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 0.5992269839080481
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9991927083333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.8025633234284988
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9993098958333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.9622032462343677
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        }
      }
    },
    {
      "id": "dex_hand",
      "label": "灵巧手采集",
      "sequence": "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003",
      "frame_index": 156,
      "original_rgb": "assets/dex_hand/original_rgb.jpg",
      "original_depth": "assets/dex_hand/original_depth.jpg",
      "depth_method": "lingbot_v05_sensor_fused",
      "samples": {
        "notransform": {
          "augmented_rgb": "assets/dex_hand/notransform/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/notransform/paired_depth.jpg",
          "rgb_changed_fraction": 0.0,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "brightness": {
          "augmented_rgb": "assets/dex_hand/brightness/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/brightness/paired_depth.jpg",
          "rgb_changed_fraction": 0.9991579861111112,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            2,
            0,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.6390843697469807
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999479166666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.4081604425175271
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9975390625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.6945956563611684
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999869791666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "contrast": {
          "augmented_rgb": "assets/dex_hand/contrast/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/contrast/paired_depth.jpg",
          "rgb_changed_fraction": 0.9998003472222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            1,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.7211064393622605
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9995833333333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.5731007750617215
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998828125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.6403516717335903
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999348958333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "saturation": {
          "augmented_rgb": "assets/dex_hand/saturation/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/saturation/paired_depth.jpg",
          "rgb_changed_fraction": 0.99015625,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            4,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.5550468028121969
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9957552083333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 1.2534268443018215
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9956119791666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 1.4826679789439483
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9791015625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "hue": {
          "augmented_rgb": "assets/dex_hand/hue/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/hue/paired_depth.jpg",
          "rgb_changed_fraction": 0.9783723958333334,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            1,
            2
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.03586181407843658
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9923567708333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.04194296459182299
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9949088541666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.030758422788418213
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9478515625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "sharpness": {
          "augmented_rgb": "assets/dex_hand/sharpness/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/sharpness/paired_depth.jpg",
          "rgb_changed_fraction": 0.4906423611111111,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 1.2111899869126033
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.4491276041666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.6405598344167173
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.6010807291666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7728805718278721
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.42171875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "random_mask": {
          "augmented_rgb": "assets/dex_hand/random_mask/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/random_mask/paired_depth.jpg",
          "rgb_changed_fraction": 0.01,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.01,
          "intended_spatial_mask_fraction": 0.01,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 141,
                "left": 38,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_l",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 62,
                "left": 105,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_r",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 84,
                "left": 283,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            }
          ]
        },
        "random_border_cutout": {
          "augmented_rgb": "assets/dex_hand/random_border_cutout/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/random_border_cutout/paired_depth.jpg",
          "rgb_changed_fraction": 0.1499956597222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.15,
          "intended_spatial_mask_fraction": 0.15,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.14998697916666667,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_l",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "top",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_r",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "top",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            }
          ]
        },
        "gaussian_noise": {
          "augmented_rgb": "assets/dex_hand/gaussian_noise/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/gaussian_noise/paired_depth.jpg",
          "rgb_changed_fraction": 0.9997526041666668,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 710800027
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998828125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 1446583567
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.999453125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 10812160
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.999921875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "gamma_correction": {
          "augmented_rgb": "assets/dex_hand/gamma_correction/augmented_rgb.jpg",
          "paired_depth": "assets/dex_hand/gamma_correction/paired_depth.jpg",
          "rgb_changed_fraction": 0.9987065972222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 0.6863960419901434
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9995833333333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.609370857062416
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9965755208333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.9623142232790163
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999609375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        }
      }
    },
    {
      "id": "chengzhong",
      "label": "称重",
      "sequence": "chengzhong_xianxia_main1",
      "frame_index": 86,
      "original_rgb": "assets/chengzhong/original_rgb.jpg",
      "original_depth": "assets/chengzhong/original_depth.jpg",
      "depth_method": "lingbot_v05_sensor_fused",
      "samples": {
        "notransform": {
          "augmented_rgb": "assets/chengzhong/notransform/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/notransform/paired_depth.jpg",
          "rgb_changed_fraction": 0.0,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "brightness": {
          "augmented_rgb": "assets/chengzhong/brightness/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/brightness/paired_depth.jpg",
          "rgb_changed_fraction": 0.9997395833333335,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            1,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.6667647233297874
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9998307291666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.8307946575298686
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.475719341304376
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9993880208333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "contrast": {
          "augmented_rgb": "assets/chengzhong/contrast/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/contrast/paired_depth.jpg",
          "rgb_changed_fraction": 0.9992621527777777,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.6177979285012178
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9994661458333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.1666279120883167
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9986848958333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.315179302278206
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9996354166666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "saturation": {
          "augmented_rgb": "assets/chengzhong/saturation/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/saturation/paired_depth.jpg",
          "rgb_changed_fraction": 0.9894661458333333,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            4
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.6438387184228958
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9714583333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.6332932212050888
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9978645833333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.5840773759293217
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9990755208333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "hue": {
          "augmented_rgb": "assets/chengzhong/hue/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/hue/paired_depth.jpg",
          "rgb_changed_fraction": 0.9821354166666666,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            1,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.04196853375760126
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9603515625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.035355979102325866
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9968359375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.03589890870939943
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.98921875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "sharpness": {
          "augmented_rgb": "assets/chengzhong/sharpness/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/sharpness/paired_depth.jpg",
          "rgb_changed_fraction": 0.4927430555555556,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 1.3065732672759593
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.56046875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.6077001833463298
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.6177083333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7420233972349501
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.30005208333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "random_mask": {
          "augmented_rgb": "assets/chengzhong/random_mask/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/random_mask/paired_depth.jpg",
          "rgb_changed_fraction": 0.01,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.01,
          "intended_spatial_mask_fraction": 0.01,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 123,
                "left": 10,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_l",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 128,
                "left": 173,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_r",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 103,
                "left": 156,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            }
          ]
        },
        "random_border_cutout": {
          "augmented_rgb": "assets/chengzhong/random_border_cutout/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/random_border_cutout/paired_depth.jpg",
          "rgb_changed_fraction": 0.14997829861111112,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.15,
          "intended_spatial_mask_fraction": 0.15,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "left",
                "pixels": 48
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_l",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "left",
                "pixels": 48
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_r",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "right",
                "pixels": 48
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.14993489583333333,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            }
          ]
        },
        "gaussian_noise": {
          "augmented_rgb": "assets/chengzhong/gaussian_noise/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/gaussian_noise/paired_depth.jpg",
          "rgb_changed_fraction": 0.9991840277777776,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 2695988850
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9980989583333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 536671702
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9996744791666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 856719944
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9997786458333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "gamma_correction": {
          "augmented_rgb": "assets/chengzhong/gamma_correction/augmented_rgb.jpg",
          "paired_depth": "assets/chengzhong/gamma_correction/paired_depth.jpg",
          "rgb_changed_fraction": 0.9942274305555555,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.8816866331475328
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9846354166666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.6937697822722493
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9984765625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.6679988929056704
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9995703125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        }
      }
    },
    {
      "id": "dajian",
      "label": "大件搬运",
      "sequence": "dajian_xianxia_main1",
      "frame_index": 232,
      "original_rgb": "assets/dajian/original_rgb.jpg",
      "original_depth": "assets/dajian/original_depth.jpg",
      "depth_method": "lingbot_v05_sensor_fused",
      "samples": {
        "notransform": {
          "augmented_rgb": "assets/dajian/notransform/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/notransform/paired_depth.jpg",
          "rgb_changed_fraction": 0.0,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "brightness": {
          "augmented_rgb": "assets/dajian/brightness/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/brightness/paired_depth.jpg",
          "rgb_changed_fraction": 0.9994574652777777,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.3915815505527531
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9989583333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.2131296363678086
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9994140625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.5148869842485532
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "contrast": {
          "augmented_rgb": "assets/dajian/contrast/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/contrast/paired_depth.jpg",
          "rgb_changed_fraction": 0.9928472222222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.3665970060024937
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9943619791666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.8040968653014701
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9957421875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.2579260124326352
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9884375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "saturation": {
          "augmented_rgb": "assets/dajian/saturation/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/saturation/paired_depth.jpg",
          "rgb_changed_fraction": 0.9419184027777779,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 1.401003550734946
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9380078125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.5234990658909705
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9300390625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.6937501130546412
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9577083333333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "hue": {
          "augmented_rgb": "assets/dajian/hue/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/hue/paired_depth.jpg",
          "rgb_changed_fraction": 0.8860503472222222,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            3
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.027810918767659655
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.8544661458333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.03440782887763086
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.856484375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.02524551400718507
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9472005208333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "sharpness": {
          "augmented_rgb": "assets/dajian/sharpness/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/sharpness/paired_depth.jpg",
          "rgb_changed_fraction": 0.4989583333333332,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            2,
            2,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 1.225513838404368
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.5044270833333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 1.2542132452600048
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.5284895833333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7898721621190822
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.4639583333333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "random_mask": {
          "augmented_rgb": "assets/dajian/random_mask/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/random_mask/paired_depth.jpg",
          "rgb_changed_fraction": 0.01,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.01,
          "intended_spatial_mask_fraction": 0.01,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 5,
                "left": 24,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_l",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 158,
                "left": 52,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_r",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 209,
                "left": 130,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            }
          ]
        },
        "random_border_cutout": {
          "augmented_rgb": "assets/dajian/random_border_cutout/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/random_border_cutout/paired_depth.jpg",
          "rgb_changed_fraction": 0.14999131944444444,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.15,
          "intended_spatial_mask_fraction": 0.15,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.14997395833333332,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_l",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_r",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            }
          ]
        },
        "gaussian_noise": {
          "augmented_rgb": "assets/dajian/gaussian_noise/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/gaussian_noise/paired_depth.jpg",
          "rgb_changed_fraction": 0.9993359375,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 866944287
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9997916666666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 1822083937
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9997916666666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 3559526139
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9984244791666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "gamma_correction": {
          "augmented_rgb": "assets/dajian/gamma_correction/augmented_rgb.jpg",
          "paired_depth": "assets/dajian/gamma_correction/paired_depth.jpg",
          "rgb_changed_fraction": 0.9947395833333333,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.511162739931788
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9986328125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.3407495477443874
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.999296875,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.9405191747891766
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9862890625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        }
      }
    },
    {
      "id": "zhoumian",
      "label": "桌面整理",
      "sequence": "zhoumian_xianxia_main1",
      "frame_index": 160,
      "original_rgb": "assets/zhoumian/original_rgb.jpg",
      "original_depth": "assets/zhoumian/original_depth.jpg",
      "depth_method": "lingbot_v05_sensor_fused",
      "samples": {
        "notransform": {
          "augmented_rgb": "assets/zhoumian/notransform/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/notransform/paired_depth.jpg",
          "rgb_changed_fraction": 0.0,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "notransform",
              "transform_type": "Identity",
              "params": {},
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "brightness": {
          "augmented_rgb": "assets/zhoumian/brightness/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/brightness/paired_depth.jpg",
          "rgb_changed_fraction": 0.9996180555555556,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 1.2755277070998323
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9988932291666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.5215747297829678
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9999609375,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "brightness",
              "transform_type": "ColorJitter",
              "params": {
                "brightness": 0.5908769713265103
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 1.0,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "contrast": {
          "augmented_rgb": "assets/zhoumian/contrast/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/contrast/paired_depth.jpg",
          "rgb_changed_fraction": 0.9969618055555555,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            2,
            1,
            2
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.8022595292792252
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9967578125,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 0.6393797359830569
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9990625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "contrast",
              "transform_type": "ColorJitter",
              "params": {
                "contrast": 1.3348446524726219
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9950651041666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "saturation": {
          "augmented_rgb": "assets/zhoumian/saturation/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/saturation/paired_depth.jpg",
          "rgb_changed_fraction": 0.9750217013888888,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            1,
            0,
            1
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.630075207810584
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9711067708333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 1.4170284222964695
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9631119791666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "saturation",
              "transform_type": "ColorJitter",
              "params": {
                "saturation": 0.7432822106327708
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9908463541666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "hue": {
          "augmented_rgb": "assets/zhoumian/hue/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/hue/paired_depth.jpg",
          "rgb_changed_fraction": 0.9464496527777778,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            3
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.0375827722044094
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9515755208333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": -0.04797341563964372
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9025651041666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "hue",
              "transform_type": "ColorJitter",
              "params": {
                "hue": 0.03354760390642629
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9852083333333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "sharpness": {
          "augmented_rgb": "assets/zhoumian/sharpness/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/sharpness/paired_depth.jpg",
          "rgb_changed_fraction": 0.4731944444444445,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.7492208410873343
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.498515625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 1.1943642062572044
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.5110677083333334,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "sharpness",
              "transform_type": "SharpnessJitter",
              "params": {
                "sharpness": 0.5166913434097108
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.41,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "random_mask": {
          "augmented_rgb": "assets/zhoumian/random_mask/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/random_mask/paired_depth.jpg",
          "rgb_changed_fraction": 0.009995659722222223,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.01,
          "intended_spatial_mask_fraction": 0.01,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 160,
                "left": 219,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.009986979166666667,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_l",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 212,
                "left": 139,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            },
            {
              "camera": "cam_r",
              "name": "random_mask",
              "transform_type": "RandomMask",
              "params": {
                "top": 66,
                "left": 269,
                "height": 24,
                "width": 32
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.01,
              "depth_changed_fraction": 0.01,
              "spatial_mask_fraction": 0.01
            }
          ]
        },
        "random_border_cutout": {
          "augmented_rgb": "assets/zhoumian/random_border_cutout/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/random_border_cutout/paired_depth.jpg",
          "rgb_changed_fraction": 0.14998697916666667,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.15,
          "intended_spatial_mask_fraction": 0.15,
          "paired_mask_iou": 1.0,
          "rgb_only_cross_modal_mismatch": true,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.1499609375,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_l",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            },
            {
              "camera": "cam_r",
              "name": "random_border_cutout",
              "transform_type": "RandomBorderCutout",
              "params": {
                "side": "bottom",
                "pixels": 36
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "paired_invalid_mask",
              "rgb_changed_fraction": 0.15,
              "depth_changed_fraction": 0.15,
              "spatial_mask_fraction": 0.15
            }
          ]
        },
        "gaussian_noise": {
          "augmented_rgb": "assets/zhoumian/gaussian_noise/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/gaussian_noise/paired_depth.jpg",
          "rgb_changed_fraction": 0.9994661458333333,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            0,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 3174660278
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.999765625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 914223746
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9994401041666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gaussian_noise",
              "transform_type": "GaussianNoise",
              "params": {
                "mean": 0.0,
                "std": 0.05,
                "noise_seed": 3691303161
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9991927083333333,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        },
        "gamma_correction": {
          "augmented_rgb": "assets/zhoumian/gamma_correction/augmented_rgb.jpg",
          "paired_depth": "assets/zhoumian/gamma_correction/paired_depth.jpg",
          "rgb_changed_fraction": 0.9966059027777777,
          "rgb_only_depth_changed_fraction": 0.0,
          "paired_depth_changed_fraction": 0.0,
          "intended_spatial_mask_fraction": 0.0,
          "paired_mask_iou": null,
          "rgb_only_cross_modal_mismatch": false,
          "trials": [
            2,
            0,
            0
          ],
          "audits": [
            {
              "camera": "cam_h",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 1.6717280237398222
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9984765625,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_l",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 0.7115156456050029
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9962760416666666,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            },
            {
              "camera": "cam_r",
              "name": "gamma_correction",
              "transform_type": "GammaCorrection",
              "params": {
                "gamma": 0.6301907899144058
              },
              "depth_mode": "paired_spatial",
              "depth_rule": "preserve_metric_depth",
              "rgb_changed_fraction": 0.9950651041666667,
              "depth_changed_fraction": 0.0,
              "spatial_mask_fraction": 0.0
            }
          ]
        }
      }
    }
  ]
};

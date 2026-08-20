window.DEPTH_QUALITY_EVIDENCE = {
  "schema": "depth_quality_evidence_v1",
  "generated_at_unix": 1787157391.1005733,
  "methods": [
    {
      "id": "raw_aligned",
      "label": "Raw aligned depth"
    },
    {
      "id": "rgb_guided",
      "label": "RGB guided"
    },
    {
      "id": "temporal_rgb_guided",
      "label": "Temporal RGB guided"
    },
    {
      "id": "lingbot_v05",
      "label": "LingBot-Depth v0.5"
    },
    {
      "id": "depth_anything_v2_fused",
      "label": "Depth Anything V2 fused"
    },
    {
      "id": "lingbot_v05_sensor_fused",
      "label": "LingBot v0.5 + sensor fusion"
    },
    {
      "id": "ai_consensus_fused",
      "label": "AI consensus fused"
    }
  ],
  "sampling": {
    "sequences": 5,
    "camera_views": 15,
    "spatial_frames": 180,
    "temporal_triplets": 90,
    "roi_frames": 410,
    "scale": 0.5,
    "seed": 20260820
  },
  "resource_guard": {
    "gpu_used": false,
    "processes": 1,
    "recommended_launcher": "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 ionice -c3 nice -n 19"
  },
  "no_model_quality": {
    "description": "Coverage, flat-region local residual, spike rate, RGB/depth edge alignment, sensor preservation, and RGB-flow-compensated temporal residual.",
    "spatial_by_method": {
      "raw_aligned": {
        "flat_region_roughness_m": 0.0003388885822561052,
        "isolated_spike_rate": 0.04205342911562916,
        "rgb_depth_edge_f1": 0.2867885187296765,
        "sensor_preservation_median_ae_m": 0.0,
        "valid_fraction": 0.6540188788434661,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0003388885822561052,
            "median": 0.0,
            "p10": 0.0,
            "p90": 9.999871253966716e-05,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.04205342911562916,
            "median": 0.011237619796098716,
            "p10": 0.0003885622293069319,
            "p90": 0.11344522078813117,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.2867885187296765,
            "median": 0.30770591723738294,
            "p10": 0.1318229210035859,
            "p90": 0.4257175326090476,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.6540188788434661,
            "median": 0.6908706761006289,
            "p10": 0.31596992924528305,
            "p90": 0.885596501572327,
            "samples": 180
          }
        }
      },
      "rgb_guided": {
        "flat_region_roughness_m": 0.0011000016497241126,
        "isolated_spike_rate": 0.040713196557609056,
        "rgb_depth_edge_f1": 0.25144089350926774,
        "sensor_preservation_median_ae_m": 0.006677773884601063,
        "valid_fraction": 0.6663099449685533,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0011000016497241126,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.00200003981590271,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.040713196557609056,
            "median": 0.009691152143864892,
            "p10": 0.000563057746788221,
            "p90": 0.10941574592474076,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.25144089350926774,
            "median": 0.2626392625083541,
            "p10": 0.11936245245813966,
            "p90": 0.3987085410177094,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.006677773884601063,
            "median": 0.0020000040531158447,
            "p10": 0.0010000020265579224,
            "p90": 0.016099810600280755,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.6663099449685533,
            "median": 0.6928557389937107,
            "p10": 0.33489681603773586,
            "p90": 0.8928193789308176,
            "samples": 180
          }
        }
      },
      "temporal_rgb_guided": {
        "flat_region_roughness_m": 0.0011666649745570288,
        "isolated_spike_rate": 0.04079232980495005,
        "rgb_depth_edge_f1": 0.2519085848833981,
        "sensor_preservation_median_ae_m": 0.007111126763953103,
        "valid_fraction": 0.6662897449336128,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0011666649745570288,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.002999994158744812,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.04079232980495005,
            "median": 0.009866275516942595,
            "p10": 0.0005228430389861402,
            "p90": 0.10984976642815703,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.2519085848833981,
            "median": 0.26234782778786514,
            "p10": 0.119702050455236,
            "p90": 0.3977227031602579,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.007111126763953103,
            "median": 0.003000020980834961,
            "p10": 0.0010001659393310547,
            "p90": 0.01610002219676971,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.6662897449336128,
            "median": 0.6928557389937107,
            "p10": 0.33489681603773586,
            "p90": 0.8928193789308176,
            "samples": 180
          }
        }
      },
      "lingbot_v05": {
        "flat_region_roughness_m": 0.0005999984012709723,
        "isolated_spike_rate": 0.0013781281777144672,
        "rgb_depth_edge_f1": 0.4370790063231823,
        "sensor_preservation_median_ae_m": 0.024266667291522027,
        "valid_fraction": 0.9934243972746332,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0005999984012709723,
            "median": 0.0009999871253967285,
            "p10": 0.0,
            "p90": 0.0010000020265579224,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.0013781281777144672,
            "median": 0.00048138302056592954,
            "p10": 8.910153281325212e-05,
            "p90": 0.00377293711906865,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.4370790063231823,
            "median": 0.41403155476107545,
            "p10": 0.3038628767798387,
            "p90": 0.5788900141200881,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.024266667291522027,
            "median": 0.011000007390975952,
            "p10": 0.004999995231628418,
            "p90": 0.06710000783205032,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.9934243972746332,
            "median": 0.9999164701257861,
            "p10": 0.9720489386792452,
            "p90": 1.0,
            "samples": 180
          }
        }
      },
      "depth_anything_v2_fused": {
        "flat_region_roughness_m": 0.00013888817694452074,
        "isolated_spike_rate": 0.03887584422290442,
        "rgb_depth_edge_f1": 0.31555237755888443,
        "sensor_preservation_median_ae_m": 0.0021444499906566406,
        "valid_fraction": 0.8940204074947591,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.00013888817694452074,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.03887584422290442,
            "median": 0.012283382230594463,
            "p10": 0.0023843833510159116,
            "p90": 0.10477575174408305,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.31555237755888443,
            "median": 0.32218803445814714,
            "p10": 0.17505479082318218,
            "p90": 0.4391165575483311,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.0021444499906566406,
            "median": 0.0010000020265579224,
            "p10": 0.0,
            "p90": 0.003999948501586914,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.8940204074947591,
            "median": 1.0,
            "p10": 0.6008529874213836,
            "p90": 1.0,
            "samples": 180
          }
        }
      },
      "lingbot_v05_sensor_fused": {
        "flat_region_roughness_m": 0.0004944427145851983,
        "isolated_spike_rate": 0.025613874425071625,
        "rgb_depth_edge_f1": 0.32933956846395823,
        "sensor_preservation_median_ae_m": 0.00229445132944319,
        "valid_fraction": 0.9947026773235501,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0004944427145851983,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0010000020265579224,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.025613874425071625,
            "median": 0.009416768046131313,
            "p10": 0.0013530330841883401,
            "p90": 0.07059057714849304,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.32933956846395823,
            "median": 0.337140213964353,
            "p10": 0.20847484866948496,
            "p90": 0.45286190666754234,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.00229445132944319,
            "median": 0.0010000020265579224,
            "p10": 0.0,
            "p90": 0.003999948501586914,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.9947026773235501,
            "median": 0.9999213836477987,
            "p10": 0.9758972091194968,
            "p90": 1.0,
            "samples": 180
          }
        }
      },
      "ai_consensus_fused": {
        "flat_region_roughness_m": 0.0002833308859003915,
        "isolated_spike_rate": 0.036151925939243656,
        "rgb_depth_edge_f1": 0.31553938557789984,
        "sensor_preservation_median_ae_m": 0.0020611136323875853,
        "valid_fraction": 0.8673410202655486,
        "distribution": {
          "flat_region_roughness_m": {
            "mean": 0.0002833308859003915,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0009999871253967285,
            "samples": 180
          },
          "isolated_spike_rate": {
            "mean": 0.036151925939243656,
            "median": 0.01004903096128518,
            "p10": 0.0014451076934909395,
            "p90": 0.0962280326661243,
            "samples": 180
          },
          "rgb_depth_edge_f1": {
            "mean": 0.31553938557789984,
            "median": 0.3371415590533737,
            "p10": 0.17975072379354867,
            "p90": 0.44396588728644215,
            "samples": 180
          },
          "sensor_preservation_median_ae_m": {
            "mean": 0.0020611136323875853,
            "median": 0.0010000020265579224,
            "p10": 0.0,
            "p90": 0.003000020980834961,
            "samples": 180
          },
          "valid_fraction": {
            "mean": 0.8673410202655486,
            "median": 0.9568396226415095,
            "p10": 0.5879107704402516,
            "p90": 0.9920990566037735,
            "samples": 180
          }
        }
      }
    },
    "temporal_by_method": {
      "raw_aligned": {
        "flow_compensated_median_residual_m": 0.01124444959892167,
        "flow_compensated_p90_residual_m": 0.20730220700303714,
        "temporal_evaluable_fraction": 0.618648890635919,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.01124444959892167,
            "median": 0.0020000040531158447,
            "p10": 0.0010000020265579224,
            "p90": 0.010099992156028756,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.20730220700303714,
            "median": 0.023000001907348633,
            "p10": 0.0070000126957893375,
            "p90": 0.6972000122070315,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.618648890635919,
            "median": 0.6508893474842767,
            "p10": 0.24985652515723272,
            "p90": 0.8835465801886793,
            "samples": 90
          }
        }
      },
      "rgb_guided": {
        "flow_compensated_median_residual_m": 0.012366681463188596,
        "flow_compensated_p90_residual_m": 0.2170832688609759,
        "temporal_evaluable_fraction": 0.6332135525856044,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.012366681463188596,
            "median": 0.002000093460083008,
            "p10": 0.001999962329864502,
            "p90": 0.01740001738071445,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.2170832688609759,
            "median": 0.02649998664855957,
            "p10": 0.012699997425079346,
            "p90": 0.743399786949158,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.6332135525856044,
            "median": 0.6637283805031446,
            "p10": 0.2591833726415094,
            "p90": 0.8911124213836479,
            "samples": 90
          }
        }
      },
      "temporal_rgb_guided": {
        "flow_compensated_median_residual_m": 0.01063333766327964,
        "flow_compensated_p90_residual_m": 0.21276000026199554,
        "temporal_evaluable_fraction": 0.6331868011879804,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.01063333766327964,
            "median": 0.001999974250793457,
            "p10": 0.0010000020265579224,
            "p90": 0.01129997372627261,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.21276000026199554,
            "median": 0.015999972820281982,
            "p10": 0.008900055289268493,
            "p90": 0.7433997750282291,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.6331868011879804,
            "median": 0.6637283805031446,
            "p10": 0.2591764937106918,
            "p90": 0.8911124213836479,
            "samples": 90
          }
        }
      },
      "lingbot_v05": {
        "flow_compensated_median_residual_m": 0.0037222274475627475,
        "flow_compensated_p90_residual_m": 0.061588882323768405,
        "temporal_evaluable_fraction": 0.9861886355695318,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.0037222274475627475,
            "median": 0.0029999911785125732,
            "p10": 0.0009999871253967285,
            "p90": 0.006999999284744263,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.061588882323768405,
            "median": 0.014499902725219727,
            "p10": 0.007999897003173828,
            "p90": 0.1696001529693604,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.9861886355695318,
            "median": 0.9965801886792454,
            "p10": 0.9506338443396227,
            "p90": 0.9998250786163522,
            "samples": 90
          }
        }
      },
      "depth_anything_v2_fused": {
        "flow_compensated_median_residual_m": 0.011322226540909874,
        "flow_compensated_p90_residual_m": 0.20995997885862985,
        "temporal_evaluable_fraction": 0.8686661425576521,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.011322226540909874,
            "median": 0.0025000423192977905,
            "p10": 0.001000046730041504,
            "p90": 0.009100002795457848,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.20995997885862985,
            "median": 0.03000006079673767,
            "p10": 0.015000012516975404,
            "p90": 0.6924000978469854,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.8686661425576521,
            "median": 0.9928164308176101,
            "p10": 0.5436979166666667,
            "p90": 0.9999135220125787,
            "samples": 90
          }
        }
      },
      "lingbot_v05_sensor_fused": {
        "flow_compensated_median_residual_m": 0.004477776752577887,
        "flow_compensated_p90_residual_m": 0.1725599906510777,
        "temporal_evaluable_fraction": 0.9876658586652692,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.004477776752577887,
            "median": 0.00299999862909317,
            "p10": 0.0019998550415039062,
            "p90": 0.00799998790025711,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.1725599906510777,
            "median": 0.029000043869018555,
            "p10": 0.015999999642372132,
            "p90": 0.5064999580383303,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.9876658586652692,
            "median": 0.9967030267295598,
            "p10": 0.9597543238993711,
            "p90": 0.9998250786163522,
            "samples": 90
          }
        }
      },
      "ai_consensus_fused": {
        "flow_compensated_median_residual_m": 0.010000000728501214,
        "flow_compensated_p90_residual_m": 0.1834210822151767,
        "temporal_evaluable_fraction": 0.8414014456673656,
        "distribution": {
          "flow_compensated_median_residual_m": {
            "mean": 0.010000000728501214,
            "median": 0.00200006365776062,
            "p10": 0.001000046730041504,
            "p90": 0.006999999284744263,
            "samples": 90
          },
          "flow_compensated_p90_residual_m": {
            "mean": 0.1834210822151767,
            "median": 0.02499993145465851,
            "p10": 0.013899999856948854,
            "p90": 0.6924000978469854,
            "samples": 90
          },
          "temporal_evaluable_fraction": {
            "mean": 0.8414014456673656,
            "median": 0.950432389937107,
            "p10": 0.5268543632075472,
            "p90": 0.9873132861635221,
            "samples": 90
          }
        }
      }
    },
    "rankings": {
      "coverage": [
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "rgb_guided",
        "temporal_rgb_guided",
        "raw_aligned"
      ],
      "flat_region_smoothness": [
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "raw_aligned",
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "rgb_guided",
        "temporal_rgb_guided"
      ],
      "edge_alignment": [
        "lingbot_v05",
        "lingbot_v05_sensor_fused",
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "raw_aligned",
        "temporal_rgb_guided",
        "rgb_guided"
      ],
      "temporal_stability": [
        "lingbot_v05",
        "lingbot_v05_sensor_fused",
        "ai_consensus_fused",
        "temporal_rgb_guided",
        "raw_aligned",
        "depth_anything_v2_fused",
        "rgb_guided"
      ]
    }
  },
  "occlusion_recovery": {
    "description": "Natural raw-depth holes evaluated only where previous and next raw frames agree after RGB optical-flow warping.",
    "by_method": {
      "raw_aligned": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 0,
        "recovery_coverage": 0.0,
        "mae_m": null,
        "rmse_m": null,
        "within_5cm_fraction": null,
        "within_10cm_fraction": null
      },
      "rgb_guided": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 38328,
        "recovery_coverage": 0.4309325178206022,
        "mae_m": 0.06953126004700919,
        "rmse_m": 0.36232458970472264,
        "within_5cm_fraction": 0.8210968482571488,
        "within_10cm_fraction": 0.9131444374869547
      },
      "temporal_rgb_guided": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 38328,
        "recovery_coverage": 0.4309325178206022,
        "mae_m": 0.06775934378976328,
        "rmse_m": 0.3622184717245702,
        "within_5cm_fraction": 0.8276194948862451,
        "within_10cm_fraction": 0.913640158630766
      },
      "lingbot_v05": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 88631,
        "recovery_coverage": 0.9965033392547953,
        "mae_m": 0.10009606626906598,
        "rmse_m": 0.4438242329850049,
        "within_5cm_fraction": 0.80975053875055,
        "within_10cm_fraction": 0.8877932100506595
      },
      "depth_anything_v2_fused": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 79450,
        "recovery_coverage": 0.8932787659373524,
        "mae_m": 0.10070566258682179,
        "rmse_m": 0.3758600894618252,
        "within_5cm_fraction": 0.6993958464443046,
        "within_10cm_fraction": 0.8010069225928257
      },
      "lingbot_v05_sensor_fused": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 88692,
        "recovery_coverage": 0.9971891794652695,
        "mae_m": 0.0878090966025487,
        "rmse_m": 0.40426609993646545,
        "within_5cm_fraction": 0.8348329048843187,
        "within_10cm_fraction": 0.9011861272719073
      },
      "ai_consensus_fused": {
        "frames_with_evidence": 90,
        "evaluable_hole_pixels": 88942,
        "recovered_pixels": 73913,
        "recovery_coverage": 0.8310247127341414,
        "mae_m": 0.05890421460430786,
        "rmse_m": 0.275407644525419,
        "within_5cm_fraction": 0.816703421590248,
        "within_10cm_fraction": 0.9165505391473759
      }
    },
    "rankings": {
      "within_5cm": [
        "lingbot_v05_sensor_fused",
        "temporal_rgb_guided",
        "rgb_guided",
        "ai_consensus_fused",
        "lingbot_v05",
        "depth_anything_v2_fused"
      ],
      "recovery_coverage": [
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "rgb_guided",
        "temporal_rgb_guided",
        "raw_aligned"
      ],
      "mae": [
        "ai_consensus_fused",
        "temporal_rgb_guided",
        "rgb_guided",
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "depth_anything_v2_fused"
      ]
    }
  },
  "multiview_geometry": {
    "strict_calibrated_reprojection": false,
    "primary_mode": "dominant-surface planarity and residual spread over synchronized head/left-wrist/right-wrist depth",
    "secondary_mode": "synchronized RGB feature correspondences + raw-depth per-frame rigid fit proxy when enough overlap exists",
    "limitation": "The bags expose per-camera internal TF but no common head/left-wrist/right-wrist robot-frame extrinsics. Planarity is rotation-invariant; feature scores are proxy residuals, not calibrated camera reprojection error.",
    "diagnostics": {
      "planarity_accepted": 210,
      "pair_attempts": 90,
      "fundamental_inliers": 16,
      "feature_rejected": 89,
      "raw_depth_rejected": 1
    },
    "planarity_by_method": {
      "raw_aligned": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.5257849542430599,
        "mean_plane_rmse_m": 0.007436887572092422,
        "three_view_plane_rmse_std_m": 0.0016965382417637163,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.5257849542430599,
            "median": 0.5526785714285715,
            "p10": 0.3769880952380953,
            "p90": 0.632805760860382,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007436887572092422,
            "median": 0.007695787177264624,
            "p10": 0.0049999165711794545,
            "p90": 0.009360873991211618,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0016965382417637163,
            "median": 0.001663923510260048,
            "p10": 0.0006980734203258081,
            "p90": 0.002518616586361391,
            "samples": 30
          }
        }
      },
      "rgb_guided": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.501312980358649,
        "mean_plane_rmse_m": 0.007428939428111823,
        "three_view_plane_rmse_std_m": 0.0013488636935876236,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.501312980358649,
            "median": 0.5263690476190476,
            "p10": 0.36026864728192165,
            "p90": 0.6101076965669989,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007428939428111823,
            "median": 0.007658050534306695,
            "p10": 0.005491509220701863,
            "p90": 0.008698580582344443,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0013488636935876236,
            "median": 0.0012401881004398371,
            "p10": 0.0004987384243297228,
            "p90": 0.0025027085428131887,
            "samples": 30
          }
        }
      },
      "temporal_rgb_guided": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.5030980186307726,
        "mean_plane_rmse_m": 0.007417843371455444,
        "three_view_plane_rmse_std_m": 0.0014611596297229499,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.5030980186307726,
            "median": 0.5304166666666668,
            "p10": 0.36554761904761907,
            "p90": 0.6143171378140918,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007417843371455444,
            "median": 0.007624722827295224,
            "p10": 0.005903302517121985,
            "p90": 0.008786936864635811,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0014611596297229499,
            "median": 0.00141948791927485,
            "p10": 0.0006016745187472749,
            "p90": 0.002443628200095123,
            "samples": 30
          }
        }
      },
      "lingbot_v05": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.5174761904761904,
        "mean_plane_rmse_m": 0.006896013966100283,
        "three_view_plane_rmse_std_m": 0.001642037447115546,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.5174761904761904,
            "median": 0.5283928571428571,
            "p10": 0.3908690476190476,
            "p90": 0.6392142857142857,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.006896013966100283,
            "median": 0.006891455616654267,
            "p10": 0.005630278587429892,
            "p90": 0.008131863814808894,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.001642037447115546,
            "median": 0.0017245126894140216,
            "p10": 0.0005592415084603003,
            "p90": 0.002492858572306808,
            "samples": 30
          }
        }
      },
      "depth_anything_v2_fused": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.46732562468758115,
        "mean_plane_rmse_m": 0.007427581328039946,
        "three_view_plane_rmse_std_m": 0.0015050493354461596,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.46732562468758115,
            "median": 0.47055778445078933,
            "p10": 0.3844404761904762,
            "p90": 0.5617857142857142,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007427581328039946,
            "median": 0.007523202753415207,
            "p10": 0.005528842567114699,
            "p90": 0.008916781102836982,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0015050493354461596,
            "median": 0.0015062402936387531,
            "p10": 0.0009367485361312108,
            "p90": 0.0021771716039904834,
            "samples": 30
          }
        }
      },
      "lingbot_v05_sensor_fused": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.48679365079365083,
        "mean_plane_rmse_m": 0.007326697619443632,
        "three_view_plane_rmse_std_m": 0.0016854558060048791,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.48679365079365083,
            "median": 0.4858928571428572,
            "p10": 0.38163095238095235,
            "p90": 0.5934166666666667,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007326697619443632,
            "median": 0.0070920519926043375,
            "p10": 0.005750644565170296,
            "p90": 0.008939131955881755,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0016854558060048791,
            "median": 0.0017162693907434678,
            "p10": 0.0006740091240214352,
            "p90": 0.002727844699735716,
            "samples": 30
          }
        }
      },
      "ai_consensus_fused": {
        "synchronized_frames": 30,
        "mean_plane_inlier_fraction": 0.4825795066710563,
        "mean_plane_rmse_m": 0.007464786932089613,
        "three_view_plane_rmse_std_m": 0.0013831172268165823,
        "distribution": {
          "mean_plane_inlier_fraction": {
            "mean": 0.4825795066710563,
            "median": 0.49457192904729086,
            "p10": 0.38197386280778517,
            "p90": 0.5597380952380953,
            "samples": 30
          },
          "mean_plane_rmse_m": {
            "mean": 0.007464786932089613,
            "median": 0.007787811875453795,
            "p10": 0.005026739784948637,
            "p90": 0.008787371115727435,
            "samples": 30
          },
          "three_view_plane_rmse_std_m": {
            "mean": 0.0013831172268165823,
            "median": 0.0014181811739620584,
            "p10": 0.0004906629168173945,
            "p90": 0.0020959228222395573,
            "samples": 30
          }
        }
      }
    },
    "ranking_by_plane_rmse": [
      "lingbot_v05",
      "lingbot_v05_sensor_fused",
      "temporal_rgb_guided",
      "depth_anything_v2_fused",
      "rgb_guided",
      "raw_aligned",
      "ai_consensus_fused"
    ],
    "ranking_by_three_view_spread": [
      "rgb_guided",
      "ai_consensus_fused",
      "temporal_rgb_guided",
      "depth_anything_v2_fused",
      "lingbot_v05",
      "lingbot_v05_sensor_fused",
      "raw_aligned"
    ],
    "feature_rigid_fit_by_method": {
      "raw_aligned": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "rgb_guided": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "temporal_rgb_guided": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "lingbot_v05": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "depth_anything_v2_fused": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "lingbot_v05_sensor_fused": {
        "accepted_pair_frames": 0,
        "distribution": {}
      },
      "ai_consensus_fused": {
        "accepted_pair_frames": 0,
        "distribution": {}
      }
    },
    "feature_ranking_by_median_residual": []
  },
  "task_roi_geometry": {
    "description": "Event-balanced geometry inside final human-approved head and executing-wrist SAM2 masks.",
    "events": 19,
    "event_views": 38,
    "sampled_track_frames": 410,
    "by_method": {
      "raw_aligned": {
        "event_views": 38,
        "roi_valid_fraction": 0.645871183908546,
        "roi_raw_hole_fill_fraction": 0.0,
        "roi_median_depth_m": 0.3981703405432871,
        "roi_p90_p10_span_m": 0.1005482724990551,
        "roi_surface_roughness_m": 5.023858763954857e-06,
        "roi_boundary_valid_fraction": 0.612103341900077,
        "roi_boundary_contrast_m": 0.04397023067285307,
        "roi_sensor_preservation_median_ae_m": 0.0,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 0.8456651620725076,
          "roi_raw_hole_fill_fraction": 0.0,
          "roi_median_depth_m": 0.6775245935802192,
          "roi_p90_p10_span_m": 0.1304064499878744,
          "roi_surface_roughness_m": 1.0047717527909713e-05,
          "roi_boundary_valid_fraction": 0.7773010224800191,
          "roi_boundary_contrast_m": 0.022287840031360705,
          "roi_sensor_preservation_median_ae_m": 0.0
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.4460772057445845,
          "roi_raw_hole_fill_fraction": 0.0,
          "roi_median_depth_m": 0.11881608750635497,
          "roi_p90_p10_span_m": 0.07069009501023585,
          "roi_surface_roughness_m": 0.0,
          "roi_boundary_valid_fraction": 0.4469056613201348,
          "roi_boundary_contrast_m": 0.06565262131434546,
          "roi_sensor_preservation_median_ae_m": 0.0
        }
      },
      "rgb_guided": {
        "event_views": 38,
        "roi_valid_fraction": 0.7086271687755737,
        "roi_raw_hole_fill_fraction": 0.47136661335030594,
        "roi_median_depth_m": 0.4032861701899556,
        "roi_p90_p10_span_m": 0.10763857909827308,
        "roi_surface_roughness_m": 0.0002598472464391996,
        "roi_boundary_valid_fraction": 0.7014005505682125,
        "roi_boundary_contrast_m": 0.037951295811570036,
        "roi_sensor_preservation_median_ae_m": 0.003786023226177942,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 0.9500903151856633,
          "roi_raw_hole_fill_fraction": 0.7297964176733087,
          "roi_median_depth_m": 0.6806052429555007,
          "roi_p90_p10_span_m": 0.13341206853564658,
          "roi_surface_roughness_m": 0.00029117251285398847,
          "roi_boundary_valid_fraction": 0.9276432938671375,
          "roi_boundary_contrast_m": 0.02491628180818948,
          "roi_sensor_preservation_median_ae_m": 0.0029144043652444746
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.46716402236548377,
          "roi_raw_hole_fill_fraction": 0.2129368090273033,
          "roi_median_depth_m": 0.12596709742441037,
          "roi_p90_p10_span_m": 0.08186508966089957,
          "roi_surface_roughness_m": 0.0002285219800244108,
          "roi_boundary_valid_fraction": 0.4751578072692874,
          "roi_boundary_contrast_m": 0.05098630981495063,
          "roi_sensor_preservation_median_ae_m": 0.004657642087111411
        }
      },
      "temporal_rgb_guided": {
        "event_views": 38,
        "roi_valid_fraction": 0.708462891347863,
        "roi_raw_hole_fill_fraction": 0.4713257819552437,
        "roi_median_depth_m": 0.40318058637168375,
        "roi_p90_p10_span_m": 0.10748356997799415,
        "roi_surface_roughness_m": 0.0003185864939002329,
        "roi_boundary_valid_fraction": 0.7011707278807432,
        "roi_boundary_contrast_m": 0.03794993562708029,
        "roi_sensor_preservation_median_ae_m": 0.004233605453396956,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 0.9500903151856633,
          "roi_raw_hole_fill_fraction": 0.7297964176733087,
          "roi_median_depth_m": 0.6803710819574065,
          "roi_p90_p10_span_m": 0.13333000972954154,
          "roi_surface_roughness_m": 0.00032182215677810947,
          "roi_boundary_valid_fraction": 0.9276432938671375,
          "roi_boundary_contrast_m": 0.024947876197616195,
          "roi_sensor_preservation_median_ae_m": 0.0034382237613613084
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.4668354675100629,
          "roi_raw_hole_fill_fraction": 0.21285514623717852,
          "roi_median_depth_m": 0.12599009078596102,
          "roi_p90_p10_span_m": 0.0816371302264468,
          "roi_surface_roughness_m": 0.0003153508310223564,
          "roi_boundary_valid_fraction": 0.47469816189434905,
          "roi_boundary_contrast_m": 0.05095199505654438,
          "roi_sensor_preservation_median_ae_m": 0.005028987145432602
        }
      },
      "lingbot_v05": {
        "event_views": 38,
        "roi_valid_fraction": 0.9998615925627455,
        "roi_raw_hole_fill_fraction": 0.9998558165580785,
        "roi_median_depth_m": 0.40094686823697523,
        "roi_p90_p10_span_m": 0.08087098054589821,
        "roi_surface_roughness_m": 0.0007207185497377731,
        "roi_boundary_valid_fraction": 0.9991313588172509,
        "roi_boundary_contrast_m": 0.039660794331787715,
        "roi_sensor_preservation_median_ae_m": 0.018147229858224045,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 1.0,
          "roi_raw_hole_fill_fraction": 1.0,
          "roi_median_depth_m": 0.6756372181232276,
          "roi_p90_p10_span_m": 0.12103730819125177,
          "roi_surface_roughness_m": 0.0004414354959489942,
          "roi_boundary_valid_fraction": 1.0,
          "roi_boundary_contrast_m": 0.04164574930542393,
          "roi_sensor_preservation_median_ae_m": 0.012324535773828395
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.999723185125491,
          "roi_raw_hole_fill_fraction": 0.9997116331161572,
          "roi_median_depth_m": 0.12625651835072302,
          "roi_p90_p10_span_m": 0.04070465290054465,
          "roi_surface_roughness_m": 0.0010000016035265518,
          "roi_boundary_valid_fraction": 0.9982627176345017,
          "roi_boundary_contrast_m": 0.0376758393581515,
          "roi_sensor_preservation_median_ae_m": 0.023969923942619688
        }
      },
      "depth_anything_v2_fused": {
        "event_views": 38,
        "roi_valid_fraction": 0.8658332255979503,
        "roi_raw_hole_fill_fraction": 0.8304131670338096,
        "roi_median_depth_m": 0.3950568561308019,
        "roi_p90_p10_span_m": 0.09376927335612407,
        "roi_surface_roughness_m": 3.0143231010893312e-05,
        "roi_boundary_valid_fraction": 0.9035375076124326,
        "roi_boundary_contrast_m": 0.04117011674028731,
        "roi_sensor_preservation_median_ae_m": 6.210114014696911e-05,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 1.0,
          "roi_raw_hole_fill_fraction": 1.0,
          "roi_median_depth_m": 0.6727394780220345,
          "roi_p90_p10_span_m": 0.13760859911775158,
          "roi_surface_roughness_m": 6.0286462021786625e-05,
          "roi_boundary_valid_fraction": 1.0,
          "roi_boundary_contrast_m": 0.036231246360878176,
          "roi_sensor_preservation_median_ae_m": 0.0
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.7316664511959005,
          "roi_raw_hole_fill_fraction": 0.6608263340676195,
          "roi_median_depth_m": 0.11737423423956932,
          "roi_p90_p10_span_m": 0.049929947594496586,
          "roi_surface_roughness_m": 0.0,
          "roi_boundary_valid_fraction": 0.807075015224865,
          "roi_boundary_contrast_m": 0.046108987119696444,
          "roi_sensor_preservation_median_ae_m": 0.00012420228029393822
        }
      },
      "lingbot_v05_sensor_fused": {
        "event_views": 38,
        "roi_valid_fraction": 0.9998745632547581,
        "roi_raw_hole_fill_fraction": 0.9998601203717209,
        "roi_median_depth_m": 0.398093646585769,
        "roi_p90_p10_span_m": 0.0858246036787388,
        "roi_surface_roughness_m": 0.00027210228085708014,
        "roi_boundary_valid_fraction": 0.9992644491504423,
        "roi_boundary_contrast_m": 0.025198753208024966,
        "roi_sensor_preservation_median_ae_m": 7.291985155815523e-05,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 1.0,
          "roi_raw_hole_fill_fraction": 1.0,
          "roi_median_depth_m": 0.6755491295927448,
          "roi_p90_p10_span_m": 0.13026201838606663,
          "roi_surface_roughness_m": 7.942512845308589e-05,
          "roi_boundary_valid_fraction": 1.0,
          "roi_boundary_contrast_m": 0.030661952609650826,
          "roi_sensor_preservation_median_ae_m": 0.0
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.9997491265095163,
          "roi_raw_hole_fill_fraction": 0.999720240743442,
          "roi_median_depth_m": 0.1206381635787931,
          "roi_p90_p10_span_m": 0.041387188971411,
          "roi_surface_roughness_m": 0.0004647794332610744,
          "roi_boundary_valid_fraction": 0.9985288983008843,
          "roi_boundary_contrast_m": 0.01973555380639911,
          "roi_sensor_preservation_median_ae_m": 0.00014583970311631047
        }
      },
      "ai_consensus_fused": {
        "event_views": 38,
        "roi_valid_fraction": 0.8588698689220293,
        "roi_raw_hole_fill_fraction": 0.7883364815579261,
        "roi_median_depth_m": 0.3977296564877549,
        "roi_p90_p10_span_m": 0.0877200805826696,
        "roi_surface_roughness_m": 2.9465330369544757e-05,
        "roi_boundary_valid_fraction": 0.888789258750581,
        "roi_boundary_contrast_m": 0.03597754184498289,
        "roi_sensor_preservation_median_ae_m": 6.210114014696911e-05,
        "head": {
          "event_views": 19,
          "roi_valid_fraction": 0.9869447865588841,
          "roi_raw_hole_fill_fraction": 0.9173102472526932,
          "roi_median_depth_m": 0.6757590420698625,
          "roi_p90_p10_span_m": 0.12617675042386897,
          "roi_surface_roughness_m": 5.454475229436701e-05,
          "roi_boundary_valid_fraction": 0.9812014624813985,
          "roi_boundary_contrast_m": 0.030242947078270833,
          "roi_sensor_preservation_median_ae_m": 0.0
        },
        "wrist": {
          "event_views": 19,
          "roi_valid_fraction": 0.7307949512851742,
          "roi_raw_hole_fill_fraction": 0.6593627158631591,
          "roi_median_depth_m": 0.11970027090564712,
          "roi_p90_p10_span_m": 0.04926341074147024,
          "roi_surface_roughness_m": 4.385908444722493e-06,
          "roi_boundary_valid_fraction": 0.7963770550197635,
          "roi_boundary_contrast_m": 0.041712136611694955,
          "roi_sensor_preservation_median_ae_m": 0.00012420228029393822
        }
      }
    },
    "rankings": {
      "roi_coverage": [
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "rgb_guided",
        "temporal_rgb_guided",
        "raw_aligned"
      ],
      "roi_surface_smoothness": [
        "raw_aligned",
        "ai_consensus_fused",
        "depth_anything_v2_fused",
        "rgb_guided",
        "lingbot_v05_sensor_fused",
        "temporal_rgb_guided",
        "lingbot_v05"
      ],
      "roi_boundary_completeness": [
        "lingbot_v05_sensor_fused",
        "lingbot_v05",
        "depth_anything_v2_fused",
        "ai_consensus_fused",
        "rgb_guided",
        "temporal_rgb_guided",
        "raw_aligned"
      ]
    }
  },
  "representatives": [
    {
      "sequence": "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003",
      "event_id": "A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003:grasp_000_right",
      "camera": "cam_r",
      "frame_index": 90,
      "image": "representatives/A10-A15-G-S-01-TQ_03_01-4_304-leju_claw-20260512172205-200049-8c435b-v003__00_wrist_cam_r_000090.jpg"
    },
    {
      "sequence": "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003",
      "event_id": "A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003:grasp_000_right",
      "camera": "cam_r",
      "frame_index": 156,
      "image": "representatives/A10-A15-G-S-01-TQ_09_01-P4_297-dex_hand-20260629103123-49-9a5c2d-v003__00_wrist_cam_r_000156.jpg"
    },
    {
      "sequence": "chengzhong_xianxia_main1",
      "event_id": "chengzhong_xianxia_main1:grasp_000_left",
      "camera": "cam_l",
      "frame_index": 86,
      "image": "representatives/chengzhong_xianxia_main1__00_wrist_cam_l_000086.jpg"
    },
    {
      "sequence": "dajian_xianxia_main1",
      "event_id": "dajian_xianxia_main1:grasp_000_right",
      "camera": "cam_r",
      "frame_index": 232,
      "image": "representatives/dajian_xianxia_main1__00_wrist_cam_r_000232.jpg"
    },
    {
      "sequence": "zhoumian_xianxia_main1",
      "event_id": "zhoumian_xianxia_main1:grasp_000_right",
      "camera": "cam_r",
      "frame_index": 160,
      "image": "representatives/zhoumian_xianxia_main1__00_wrist_cam_r_000160.jpg"
    }
  ],
  "limitations": [
    "Raw sensor depth and temporal consensus are references, not laser-scanner ground truth.",
    "Lower roughness can indicate denoising or over-smoothing; interpret it with edge alignment and sensor preservation.",
    "Natural-hole recovery is evaluated only on temporally observable, bidirectionally consistent pixels.",
    "Cross-view planarity is rotation-invariant; calibrated reprojection remains unavailable because common robot-frame camera extrinsics are absent.",
    "No attention map or trained policy output is used in these four evidence families."
  ],
  "elapsed_seconds": 62.09814786911011
};

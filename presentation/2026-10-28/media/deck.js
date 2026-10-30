window.DECK={
 "main": [
  {
   "id": "01_problem",
   "title": "More information from fewer measurements",
   "cue": "The laboratory challenge",
   "type": "Conceptual diagram",
   "audio": "media/audio/01_problem.mp3",
   "duration": 22.96,
   "source_span": [
    10.276,
    33.239
   ],
   "cues": [
    {
     "t": 0,
     "visual": "01_problem"
    }
   ],
   "captions": [
    {
     "t0": 0.46,
     "t1": 4.66,
     "text": "When we study how a substance affects the development of a biological neural network,",
     "zh": "研究某种物质如何影响生物神经网络的发育时，我们需要测试不同浓度，并在数天内持续监测变化。"
    },
    {
     "t0": 5.12,
     "t1": 9.54,
     "text": "we need to test different concentrations and monitor changes over several days.",
     "zh": "研究某种物质如何影响生物神经网络的发育时，我们需要测试不同浓度，并在数天内持续监测变化。"
    },
    {
     "t0": 10.3,
     "t1": 13.56,
     "text": "This requires cell cultures, equipment, and time.",
     "zh": "这需要细胞培养、仪器设备和时间。"
    },
    {
     "t0": 14.2,
     "t1": 18.8,
     "text": "The question is: can we use knowledge from previous experiments to fill",
     "zh": "问题在于：能否利用以往实验的知识，补全新物质的响应，并决定下一步测量什么？"
    },
    {
     "t0": 18.8,
     "t1": 22.74,
     "text": "in the response of a new substance and decide what to measure next?",
     "zh": "问题在于：能否利用以往实验的知识，补全新物质的响应，并决定下一步测量什么？"
    }
   ],
   "script": "When we study how a substance affects the development of a biological neural network, we need to test different concentrations and monitor changes over several days. This requires cell cultures, equipment, and time. The question is: can we use knowledge from previous experiments to fill in the response of a new substance and decide what to measure next?"
  },
  {
   "id": "02_digital_twin",
   "title": "From three concentrations to a complete response",
   "cue": "How the digital twin learns across compounds",
   "type": "Conceptual diagram",
   "audio": "media/audio/02_digital_twin.mp3",
   "duration": 32.04,
   "source_span": [
    33.239,
    65.283
   ],
   "cues": [
    {
     "t": 0,
     "visual": "02_digital_twin"
    }
   ],
   "captions": [
    {
     "t0": 0.93,
     "t1": 2.9,
     "text": "NeuroChip Twin addresses this challenge through",
     "zh": "NeuroChip Twin 通过基于真实实验数据构建的数字孪生来应对这一挑战。"
    },
    {
     "t0": 2.9,
     "t1": 5.8,
     "text": "a digital twin built on real experimental data.",
     "zh": "NeuroChip Twin 通过基于真实实验数据构建的数字孪生来应对这一挑战。"
    },
    {
     "t0": 6.48,
     "t1": 10.52,
     "text": "It is designed for neurotoxicology laboratories and researchers",
     "zh": "它面向神经毒理学实验室，以及开发具有电信号读出的芯片神经系统的研究人员。"
    },
    {
     "t0": 10.52,
     "t1": 13.6,
     "text": "developing neural systems on chips with electrical readouts.",
     "zh": "它面向神经毒理学实验室，以及开发具有电信号读出的芯片神经系统的研究人员。"
    },
    {
     "t0": 14.24,
     "t1": 16.6,
     "text": "The model learns patterns shared across substances.",
     "zh": "模型学习不同物质之间的共同规律。"
    },
    {
     "t0": 17.34,
     "t1": 20.58,
     "text": "Given measurements at a few concentrations of a new compound,",
     "zh": "只需新化合物在少数几个浓度下的测量值，它就能估计研究所涵盖各时间点上其余浓度的响应。"
    },
    {
     "t0": 20.82,
     "t1": 26.46,
     "text": "it estimates the response at the remaining concentrations across the time points covered by the study.",
     "zh": "只需新化合物在少数几个浓度下的测量值，它就能估计研究所涵盖各时间点上其余浓度的响应。"
    },
    {
     "t0": 27.1,
     "t1": 31.78,
     "text": "It also provides uncertainty intervals and flags predictions with low confidence.",
     "zh": "它还给出不确定性区间，并标记置信度低的预测。"
    }
   ],
   "script": "NeuroChip Twin addresses this challenge through a digital twin built on real experimental data. It is designed for neurotoxicology laboratories and researchers developing neural systems on chips with electrical readouts. The model learns patterns shared across substances. Given measurements at a few concentrations of a new compound, it estimates the response at the remaining concentrations across the time points covered by the study. It also provides uncertainty intervals and flags predictions with low confidence."
  },
  {
   "id": "03_innovation",
   "title": "Learning that informs experimental decisions",
   "cue": "Three elements of the innovation",
   "type": "Architecture",
   "audio": "media/audio/03_innovation.mp3",
   "duration": 59.35,
   "source_span": [
    65.283,
    124.633
   ],
   "cues": [
    {
     "t": 0,
     "visual": "03_innovation"
    }
   ],
   "captions": [
    {
     "t0": 0.44,
     "t1": 2.2,
     "text": "The innovation combines three elements.",
     "zh": "其创新结合了三个要素。"
    },
    {
     "t0": 2.8,
     "t1": 4.58,
     "text": "First, learning across compounds:",
     "zh": "第一，跨化合物学习：每个新实验都能利用以往实验积累的信息，而无需该物质的完整响应曲线。"
    },
    {
     "t0": 5.24,
     "t1": 9.38,
     "text": "each new experiment benefits from information gathered in previous experiments,",
     "zh": "第一，跨化合物学习：每个新实验都能利用以往实验积累的信息，而无需该物质的完整响应曲线。"
    },
    {
     "t0": 9.38,
     "t1": 13.28,
     "text": "without requiring a complete response curve for that substance.",
     "zh": "第一，跨化合物学习：每个新实验都能利用以往实验积累的信息，而无需该物质的完整响应曲线。"
    },
    {
     "t0": 14.14,
     "t1": 18.86,
     "text": "Second, an architecture that incorporates interpolation between measured doses.",
     "zh": "第二，一种融合已测剂量之间插值的模型架构。"
    },
    {
     "t0": 18.86,
     "t1": 22.62,
     "text": "The model learns when to rely more on these observations",
     "zh": "模型会学习何时更依赖这些观测值，何时利用从其他化合物学到的规律。"
    },
    {
     "t0": 22.62,
     "t1": 25.38,
     "text": "and when to use patterns learned from other compounds.",
     "zh": "模型会学习何时更依赖这些观测值，何时利用从其他化合物学到的规律。"
    },
    {
     "t0": 26.2,
     "t1": 32.56,
     "text": "Third, a connection between prediction and experimental decisions: completing response curves,",
     "zh": "第三，将预测与实验决策相连接：补全响应曲线、估计生物活性、指导下一个剂量的选择，并探索检测何时可以提前终止。"
    },
    {
     "t0": 32.98,
     "t1": 37.12,
     "text": "estimating biological activity, guiding the choice of the next dose,",
     "zh": "第三，将预测与实验决策相连接：补全响应曲线、估计生物活性、指导下一个剂量的选择，并探索检测何时可以提前终止。"
    },
    {
     "t0": 37.54,
     "t1": 40.14,
     "text": "and investigating when an assay could stop early.",
     "zh": "第三，将预测与实验决策相连接：补全响应曲线、估计生物活性、指导下一个剂量的选择，并探索检测何时可以提前终止。"
    },
    {
     "t0": 40.88,
     "t1": 45.18,
     "text": "Compared with conventional interpolation, the system can draw on information",
     "zh": "与传统插值相比，该系统能够利用其他物质的信息以及神经网络的多项特征。"
    },
    {
     "t0": 45.18,
     "t1": 48.74,
     "text": "from other substances and multiple features of the neural network.",
     "zh": "与传统插值相比，该系统能够利用其他物质的信息以及神经网络的多项特征。"
    },
    {
     "t0": 49.5,
     "t1": 50.68,
     "text": "Compared with a Hill curve,",
     "zh": "与希尔曲线相比，它能够表示不符合该特定形状的响应。"
    },
    {
     "t0": 50.98,
     "t1": 54.42,
     "text": "it can represent responses that do not follow that particular shape.",
     "zh": "与希尔曲线相比，它能够表示不符合该特定形状的响应。"
    },
    {
     "t0": 54.88,
     "t1": 57.14,
     "text": "We evaluated these alternatives using",
     "zh": "我们在评估这些方法时，为每种方法提供了相同的可用测量数据。"
    },
    {
     "t0": 57.14,
     "t1": 59.18,
     "text": "the same available measurements for every method.",
     "zh": "我们在评估这些方法时，为每种方法提供了相同的可用测量数据。"
    }
   ],
   "script": "The innovation combines three elements. First, learning across compounds: each new experiment benefits from information gathered in previous experiments, without requiring a complete response curve for that substance. Second, an architecture that incorporates interpolation between measured doses. The model learns when to rely more on these observations and when to use patterns learned from other compounds. Third, a connection between prediction and experimental decisions: completing response curves, estimating biological activity, guiding the choice of the next dose, and investigating when an assay could stop early. Compared with conventional interpolation, the system can draw on information from other substances and multiple features of the neural network. Compared with a Hill curve, it can represent responses that do not follow that particular shape. We evaluated these alternatives using the same available measurements for every method."
  },
  {
   "id": "04_earlier_work",
   "title": "Building on earlier research",
   "cue": "Initial prototype and Speaking to Silicon",
   "type": "Earlier work",
   "audio": "media/audio/04_earlier_work.mp3",
   "duration": 27.32,
   "source_span": [
    124.633,
    151.955
   ],
   "cues": [
    {
     "t": 0,
     "visual": "04_earlier_work"
    }
   ],
   "captions": [
    {
     "t0": 0.43,
     "t1": 5.11,
     "text": "The project builds on my first NeuroChip Twin prototype, which used synthetic data.",
     "zh": "本项目建立在我的第一个 NeuroChip Twin 原型之上，该原型使用的是合成数据。"
    },
    {
     "t0": 5.63,
     "t1": 10.71,
     "text": "The submitted version uses real biological data and a new validation framework.",
     "zh": "提交的版本使用真实生物数据和新的验证框架。"
    },
    {
     "t0": 11.47,
     "t1": 15.51,
     "text": "It also draws on an idea from my earlier work, \"Speaking to Silicon\":",
     "zh": "它还借鉴了我早期工作《Speaking to Silicon》中的一个思路：利用部分信号来判断一个高成本的过程是否应继续。"
    },
    {
     "t0": 16.29,
     "t1": 20.47,
     "text": "using a partial signal to decide whether a costly process should continue.",
     "zh": "它还借鉴了我早期工作《Speaking to Silicon》中的一个思路：利用部分信号来判断一个高成本的过程是否应继续。"
    },
    {
     "t0": 21.33,
     "t1": 23.57,
     "text": "In this project, that idea inspires",
     "zh": "在本项目中，这一思路启发了对生物检测提前终止的研究。"
    },
    {
     "t0": 23.57,
     "t1": 27.07,
     "text": "the investigation of early stopping in biological assays.",
     "zh": "在本项目中，这一思路启发了对生物检测提前终止的研究。"
    }
   ],
   "script": "The project builds on my first NeuroChip Twin prototype, which used synthetic data. The submitted version uses real biological data and a new validation framework. It also draws on an idea from my earlier work, \"Speaking to Silicon\": using a partial signal to decide whether a costly process should continue. In this project, that idea inspires the investigation of early stopping in biological assays."
  },
  {
   "id": "05_data",
   "title": "Real biological data",
   "cue": "EPA data: 243 compounds, 17 features, 4 time points",
   "type": "Original figure and data",
   "audio": "media/audio/05_data.mp3",
   "duration": 12.85,
   "source_span": [
    151.955,
    164.807
   ],
   "cues": [
    {
     "t": 0,
     "visual": "05_data"
    }
   ],
   "captions": [
    {
     "t0": 0.34,
     "t1": 1.58,
     "text": "To evaluate the system,",
     "zh": "为评估该系统，我使用了美国环境保护署（EPA）的公开数据：243 种物质，在四个发育阶段测量了 17 项神经活动特征。"
    },
    {
     "t0": 2.1,
     "t1": 7.68,
     "text": "I used public data from the US Environmental Protection Agency: 243 substances,",
     "zh": "为评估该系统，我使用了美国环境保护署（EPA）的公开数据：243 种物质，在四个发育阶段测量了 17 项神经活动特征。"
    },
    {
     "t0": 7.68,
     "t1": 12.6,
     "text": "with 17 features of neural activity measured at four stages of development.",
     "zh": "为评估该系统，我使用了美国环境保护署（EPA）的公开数据：243 种物质，在四个发育阶段测量了 17 项神经活动特征。"
    }
   ],
   "script": "To evaluate the system, I used public data from the US Environmental Protection Agency: 243 substances, with 17 features of neural activity measured at four stages of development."
  },
  {
   "id": "06_validation",
   "title": "Evaluation on unseen compounds",
   "cue": "Training and evaluation separated by substance",
   "type": "Evaluation protocol",
   "audio": "media/audio/06_validation.mp3",
   "duration": 8.63,
   "source_span": [
    164.807,
    173.441
   ],
   "cues": [
    {
     "t": 0,
     "visual": "06_validation"
    }
   ],
   "captions": [
    {
     "t0": 0.29,
     "t1": 3.15,
     "text": "Training and evaluation were separated by substance.",
     "zh": "训练与评估按物质划分。"
    },
    {
     "t0": 3.15,
     "t1": 6.51,
     "text": "This allowed us to assess performance on compounds",
     "zh": "这样就能评估模型在训练中从未见过的化合物上的表现。"
    },
    {
     "t0": 6.51,
     "t1": 8.47,
     "text": "the model had not encountered during training.",
     "zh": "这样就能评估模型在训练中从未见过的化合物上的表现。"
    }
   ],
   "script": "Training and evaluation were separated by substance. This allowed us to assess performance on compounds the model had not encountered during training."
  },
  {
   "id": "07_results",
   "title": "Lower error than five baselines",
   "cue": "About 11% lower error with three concentrations",
   "type": "Evaluation result",
   "audio": "media/audio/07_results.mp3",
   "duration": 19.74,
   "source_span": [
    173.441,
    193.179
   ],
   "cues": [
    {
     "t": 0,
     "visual": "07_results"
    }
   ],
   "captions": [
    {
     "t0": 0.56,
     "t1": 4.58,
     "text": "The main result is an approximately 11% reduction in mean",
     "zh": "主要结果：在已有三个浓度测量值的情况下，与五种评估基线中最强的对数线性插值相比，平均绝对误差降低了约 11%。"
    },
    {
     "t0": 4.58,
     "t1": 9.02,
     "text": "absolute error when measurements at three concentrations are available,",
     "zh": "主要结果：在已有三个浓度测量值的情况下，与五种评估基线中最强的对数线性插值相比，平均绝对误差降低了约 11%。"
    },
    {
     "t0": 9.44,
     "t1": 15.16,
     "text": "compared with the strongest of the five evaluated baselines: log-linear interpolation.",
     "zh": "主要结果：在已有三个浓度测量值的情况下，与五种评估基线中最强的对数线性插值相比，平均绝对误差降低了约 11%。"
    },
    {
     "t0": 15.8,
     "t1": 19.5,
     "text": "The improvement remained when we excluded the partition used during development.",
     "zh": "排除开发阶段使用的数据划分后，这一改进依然存在。"
    }
   ],
   "script": "The main result is an approximately 11% reduction in mean absolute error when measurements at three concentrations are available, compared with the strongest of the five evaluated baselines: log-linear interpolation. The improvement remained when we excluded the partition used during development."
  },
  {
   "id": "08_uncertainty",
   "title": "Predictions with quantified uncertainty",
   "cue": "89.9% observed coverage for 90% intervals",
   "type": "Calibration result",
   "audio": "media/audio/08_uncertainty.mp3",
   "duration": 8.9,
   "source_span": [
    193.179,
    202.081
   ],
   "cues": [
    {
     "t": 0,
     "visual": "08_uncertainty"
    }
   ],
   "captions": [
    {
     "t0": -0.24,
     "t1": 4.6,
     "text": "In addition, prediction intervals targeting 90% coverage achieved",
     "zh": "此外，在回顾性评估中，目标覆盖率为 90% 的预测区间实际覆盖率达到 89.9%。"
    },
    {
     "t0": 5.8,
     "t1": 8.46,
     "text": "89.9% observed coverage in the retrospective evaluation.",
     "zh": "此外，在回顾性评估中，目标覆盖率为 90% 的预测区间实际覆盖率达到 89.9%。"
    }
   ],
   "script": "In addition, prediction intervals targeting 90% coverage achieved 89.9% observed coverage in the retrospective evaluation."
  },
  {
   "id": "09_resources",
   "title": "An opportunity to reduce experimental work",
   "cue": "Three of seven concentrations: 57% fewer treated wells",
   "type": "Design calculation",
   "audio": "media/audio/09_resources.mp3",
   "duration": 17.27,
   "source_span": [
    202.081,
    219.349
   ],
   "cues": [
    {
     "t": 0,
     "visual": "09_resources"
    }
   ],
   "captions": [
    {
     "t0": 0.56,
     "t1": 3.26,
     "text": "In a seven-concentration experimental design,",
     "zh": "在七个浓度的实验设计中，只测量三个浓度意味着处理孔减少约 57%。"
    },
    {
     "t0": 3.7,
     "t1": 9.38,
     "text": "measuring three concentrations represents approximately 57% fewer treated wells.",
     "zh": "在七个浓度的实验设计中，只测量三个浓度意味着处理孔减少约 57%。"
    },
    {
     "t0": 9.98,
     "t1": 13.16,
     "text": "This offers an opportunity to reduce experimental work,",
     "zh": "这为减少实验工作量提供了机会，但实际节省仍需在实验室中验证。"
    },
    {
     "t0": 13.46,
     "t1": 17.02,
     "text": "although the actual savings still need to be demonstrated in the laboratory.",
     "zh": "这为减少实验工作量提供了机会，但实际节省仍需在实验室中验证。"
    }
   ],
   "script": "In a seven-concentration experimental design, measuring three concentrations represents approximately 57% fewer treated wells. This offers an opportunity to reduce experimental work, although the actual savings still need to be demonstrated in the laboratory."
  },
  {
   "id": "10_future",
   "title": "From retrospective results to laboratory validation",
   "cue": "Current status, next study and conclusion",
   "type": "Status and future work",
   "audio": "media/audio/10_future.mp3",
   "duration": 57.1,
   "source_span": [
    219.349,
    276.446
   ],
   "cues": [
    {
     "t": 0,
     "visual": "10_future"
    }
   ],
   "captions": [
    {
     "t0": -0.25,
     "t1": 2.49,
     "text": "The solution includes trained models,",
     "zh": "该方案包括训练好的模型、可运行的演示和公开文档。"
    },
    {
     "t0": 2.99,
     "t1": 5.93,
     "text": "a runnable demonstration, and public documentation.",
     "zh": "该方案包括训练好的模型、可运行的演示和公开文档。"
    },
    {
     "t0": 6.63,
     "t1": 8.93,
     "text": "The demonstration can run on a CPU.",
     "zh": "演示可以在 CPU 上运行。"
    },
    {
     "t0": 9.69,
     "t1": 12.79,
     "text": "The next step is prospective validation with a laboratory:",
     "zh": "下一步是与实验室合作开展前瞻性验证：用系统选择测量点，再通过实验检验其预测。"
    },
    {
     "t0": 13.27,
     "t1": 17.75,
     "text": "using the system to select measurements, then testing its predictions experimentally.",
     "zh": "下一步是与实验室合作开展前瞻性验证：用系统选择测量点，再通过实验检验其预测。"
    },
    {
     "t0": 19.09,
     "t1": 22.11,
     "text": "The main evaluation uses rat neuronal cultures.",
     "zh": "主要评估使用的是大鼠神经元培养物。"
    },
    {
     "t0": 22.97,
     "t1": 25.25,
     "text": "Applying the system to a new chip platform",
     "zh": "将系统应用于新的芯片平台或人类细胞，需要重新验证和校准。"
    },
    {
     "t0": 25.25,
     "t1": 28.81,
     "text": "or to human cells will require validation and recalibration.",
     "zh": "将系统应用于新的芯片平台或人类细胞，需要重新验证和校准。"
    },
    {
     "t0": 28.81,
     "t1": 32.55,
     "text": "Looking ahead, I want to extend this approach to other platforms",
     "zh": "展望未来，我希望把这一方法推广到其他平台，并研究哪些测量相对于所需资源能提供最多的信息。"
    },
    {
     "t0": 32.55,
     "t1": 38.25,
     "text": "and investigate which measurements provide the most information relative to the resources they require.",
     "zh": "展望未来，我希望把这一方法推广到其他平台，并研究哪些测量相对于所需资源能提供最多的信息。"
    },
    {
     "t0": 39.11,
     "t1": 40.71,
     "text": "My aim is to turn existing",
     "zh": "我的目标是把现有实验数据转化为更有依据的实验决策。"
    },
    {
     "t0": 40.71,
     "t1": 44.43,
     "text": "experimental data into better-informed experimental decisions.",
     "zh": "我的目标是把现有实验数据转化为更有依据的实验决策。"
    },
    {
     "t0": 45.45,
     "t1": 49.57,
     "text": "NeuroChip Twin demonstrates a measurable improvement in completing responses",
     "zh": "NeuroChip Twin 在仅凭少数测试浓度补全响应方面取得了可衡量的改进，并为在实验室中评估其实际价值提供了开放的基础。"
    },
    {
     "t0": 49.57,
     "t1": 53.35,
     "text": "from a few tested concentrations and provides an open",
     "zh": "NeuroChip Twin 在仅凭少数测试浓度补全响应方面取得了可衡量的改进，并为在实验室中评估其实际价值提供了开放的基础。"
    },
    {
     "t0": 53.35,
     "t1": 56.61,
     "text": "foundation for evaluating its practical value in the laboratory.",
     "zh": "NeuroChip Twin 在仅凭少数测试浓度补全响应方面取得了可衡量的改进，并为在实验室中评估其实际价值提供了开放的基础。"
    }
   ],
   "script": "The solution includes trained models, a runnable demonstration, and public documentation. The demonstration can run on a CPU. The next step is prospective validation with a laboratory: using the system to select measurements, then testing its predictions experimentally. The main evaluation uses rat neuronal cultures. Applying the system to a new chip platform or to human cells will require validation and recalibration. Looking ahead, I want to extend this approach to other platforms and investigate which measurements provide the most information relative to the resources they require. My aim is to turn existing experimental data into better-informed experimental decisions. NeuroChip Twin demonstrates a measurable improvement in completing responses from a few tested concentrations and provides an open foundation for evaluating its practical value in the laboratory."
  }
 ],
 "qa": [
  {
   "id": "qa1",
   "title": "What is really new here, if interpolation and dose-response models already exist?",
   "title_zh": "既然插值和剂量-响应模型已经存在，这里真正的新意是什么？",
   "cue": "Innovation",
   "type": "Q&A answer",
   "audio": "media/audio/qa1.mp3",
   "sync": true,
   "duration": 37.43,
   "cues": [
    {
     "t": 0.0,
     "visual": "13_qa_innovation"
    }
   ],
   "captions": [
    {
     "t0": 0.0,
     "t1": 3.04,
     "text": "Conventional interpolation and curve fitting mainly",
     "zh": "传统插值和曲线拟合主要使用被测物质自身的测量数据。"
    },
    {
     "t0": 3.04,
     "t1": 5.56,
     "text": "use measurements from the substance being tested.",
     "zh": "传统插值和曲线拟合主要使用被测物质自身的测量数据。"
    },
    {
     "t0": 6.56,
     "t1": 9.7,
     "text": "NeuroChip Twin also learns shared patterns across compounds,",
     "zh": "NeuroChip Twin 还学习化合物之间的共同规律，并学习如何将这些规律与已有测量数据结合。"
    },
    {
     "t0": 10.24,
     "t1": 13.38,
     "text": "and learns how to combine these patterns with the available measurements.",
     "zh": "NeuroChip Twin 还学习化合物之间的共同规律，并学习如何将这些规律与已有测量数据结合。"
    },
    {
     "t0": 14.06,
     "t1": 17.6,
     "text": "We evaluated it on a dataset of 243 substances,",
     "zh": "我们在包含 243 种物质的数据集上进行了评估，每个测试化合物都不出现在其对应模型的训练数据中。"
    },
    {
     "t0": 18.1,
     "t1": 21.32,
     "text": "keeping each test compound out of its models' training data.",
     "zh": "我们在包含 243 种物质的数据集上进行了评估，每个测试化合物都不出现在其对应模型的训练数据中。"
    },
    {
     "t0": 21.32,
     "t1": 24.32,
     "text": "With measurements at three concentrations,",
     "zh": "在有三个浓度测量值时，平均绝对误差比最强的评估基线降低了约 11%。"
    },
    {
     "t0": 24.92,
     "t1": 29.92,
     "text": "mean absolute error fell by about 11% against the strongest evaluated baseline.",
     "zh": "在有三个浓度测量值时，平均绝对误差比最强的评估基线降低了约 11%。"
    },
    {
     "t0": 30.64,
     "t1": 34.54,
     "text": "Its calibrated prediction intervals achieved 89.9% coverage",
     "zh": "在回顾性测试中，其校准后的预测区间在 90% 目标下达到了 89.9% 的覆盖率。"
    },
    {
     "t0": 34.54,
     "t1": 37.0,
     "text": "for a 90% target in retrospective testing.",
     "zh": "在回顾性测试中，其校准后的预测区间在 90% 目标下达到了 89.9% 的覆盖率。"
    }
   ],
   "script": "Conventional interpolation and curve fitting mainly use measurements from the substance being tested. NeuroChip Twin also learns shared patterns across compounds, and learns how to combine these patterns with the available measurements. We evaluated it on a dataset of 243 substances, keeping each test compound out of its models' training data. With measurements at three concentrations, mean absolute error fell by about 11% against the strongest evaluated baseline. Its calibrated prediction intervals achieved 89.9% coverage for a 90% target in retrospective testing."
  },
  {
   "id": "qa2",
   "title": "Has it been validated in a real laboratory or on an organ-on-chip?",
   "title_zh": "它是否已在真实实验室或器官芯片上得到验证？",
   "cue": "Laboratory validation",
   "type": "Q&A answer",
   "audio": "media/audio/qa2.mp3",
   "sync": true,
   "duration": 29.68,
   "cues": [
    {
     "t": 0.0,
     "visual": "14_qa_validation"
    }
   ],
   "captions": [
    {
     "t0": 0.0,
     "t1": 2.5,
     "text": "Not yet prospectively, and I don't claim it has.",
     "zh": "尚未进行前瞻性验证，我也不声称已经完成。"
    },
    {
     "t0": 2.98,
     "t1": 8.82,
     "text": "The current validation is retrospective, on real EPA data from rat neuronal cultures,",
     "zh": "目前的验证是回顾性的，基于 EPA 大鼠神经元培养物的真实数据，训练与测试按物质划分。"
    },
    {
     "t0": 8.82,
     "t1": 11.44,
     "text": "with training and testing split by substance.",
     "zh": "目前的验证是回顾性的，基于 EPA 大鼠神经元培养物的真实数据，训练与测试按物质划分。"
    },
    {
     "t0": 12.14,
     "t1": 14.62,
     "text": "The next step is a prospective study with a laboratory:",
     "zh": "下一步是与实验室合作开展前瞻性研究：由系统选择要测量的浓度，再由实验室检验其预测。"
    },
    {
     "t0": 15.34,
     "t1": 20.4,
     "text": "the system chooses which concentrations to measure, and the lab tests its predictions.",
     "zh": "下一步是与实验室合作开展前瞻性研究：由系统选择要测量的浓度，再由实验室检验其预测。"
    },
    {
     "t0": 20.86,
     "t1": 23.18,
     "text": "That study will measure the actual savings.",
     "zh": "该研究将衡量实际节省的资源。"
    },
    {
     "t0": 23.82,
     "t1": 26.52,
     "text": "Each new chip platform or human-cell model",
     "zh": "每个新的芯片平台或人类细胞模型都需要各自的验证和重新校准。"
    },
    {
     "t0": 26.52,
     "t1": 29.22,
     "text": "will need its own validation and recalibration.",
     "zh": "每个新的芯片平台或人类细胞模型都需要各自的验证和重新校准。"
    }
   ],
   "script": "Not yet prospectively, and I don't claim it has. The current validation is retrospective, on real EPA data from rat neuronal cultures, with training and testing split by substance. The next step is a prospective study with a laboratory: the system chooses which concentrations to measure, and the lab tests its predictions. That study will measure the actual savings. Each new chip platform or human-cell model will need its own validation and recalibration."
  }
 ],
 "thanks": {
  "audio": "media/audio/thanks.mp3",
  "duration": 1.08,
  "captions": [
   {
    "t0": 0.03,
    "t1": 0.63,
    "text": "Thank you.",
    "zh": "谢谢。"
   }
  ]
 },
 "visual_titles": {
  "01_problem": "More information from fewer measurements",
  "02_digital_twin": "From three concentrations to a complete response",
  "03_innovation": "Learning that informs experimental decisions",
  "04_earlier_work": "Building on earlier research",
  "05_data": "Real biological data",
  "06_validation": "Evaluation on unseen compounds",
  "07_results": "Lower error than five baselines",
  "08_uncertainty": "Predictions with quantified uncertainty",
  "09_resources": "An opportunity to reduce experimental work",
  "10_future": "From retrospective results to laboratory validation",
  "11_real_example": "A real example: measurements, prediction, uncertainty",
  "12_real_chip": "Activity recorded from a real neural chip",
  "13_qa_innovation": "What is new: learning across compounds",
  "14_qa_validation": "Validation: retrospective today, prospective next"
 },
 "main_total": 266.16
};

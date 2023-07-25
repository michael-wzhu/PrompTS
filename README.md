# PrompTS
A repo for instruction tuning of time series data. One of the first attempts to apply LLMs into general time series studies! 

随着大语言模型(LLMs)的发展，其在各个领域都开始发挥着重要作用，如应用开发，多模态交互等。我们不得不思考：LLMs是否可以作为一个数据分析的统一接口？当然，LLMs做数据分析时，是自主驱使外接引擎/模型，还是自身直接处理数据，这是一个问题。

本项目目前先着重于后者：LLMs直接处理数据。我们采用数据分析中一个非常重要的场景，时间序列分析，作为我们的试验场地。我们会将时间序列分析中的多种不同任务类型，上百个来自不同领域的基准数据集整理为自然语言提示，形成首个大规模多任务时间序列提示数据集**PrompTS** (**Promp**ts for **T**ime **S**eries)。

在任务层面，**PrompTS** 包含以下任务和数据集(初步规划，后续将持续动态调整):
- [prompt-based time series classification](./src/data_proc/cls): 原始数据来源于[UCR-archieve](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/); 
- [prompt-based time series NLI](./src/data_proc/nli): 原始数据来源于[UCR-archieve](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)。注意在这个数据中，很多数据的分类标签是不包含实际意义的，比如类别是采集数据来源的几个不同受试者。所以我们将这类时序分类任务转化为time series inference任务(类似于NLP中的NLI任务)，即判断两段时序是否来自同一个类别/受试者。
- ⏳ [prompt-based time series forecasting](./src/data_proc/forecast): 我们目前采用与[N-Beats](https://github.com/ServiceNow/N-BEATS)和[Autoformer](https://arxiv.org/pdf/2106.13008.pdf)相同的数据集，包括Electricity, Tourism, Traffic，M3, M4, ETT, Exchange。同时我们也借鉴和扩展了[PISA](https://github.com/HaoUNSW/PISA)项目的数据集和模板。目前我们的预测任务以短时预测为主，将会逐步覆盖更具有挑战性的长时预测任务。
- ⏳ [prompt-based time series imputation](./src/data_proc/forecast): 
- ⏳ [prompt-based time series extrinsic regression](./): 数据来源于[tseregression](http://tseregression.org/).这一任务要求模型根据时间序列数据预测一个外部标量数据。很明显，这类任务的传统解法与TSC紧密相关。


----

[Chinese-LlaMA2大模型](https://github.com/michael-wzhu/Chinese-LlaMA2) | [中文医疗大模型ChatMed](https://github.com/michael-wzhu/ChatMed) |  [业内首个中医药大模型ShenNong-TCM-LLM](https://github.com/michael-wzhu/ShenNong-TCM-LLM) | [PromptCBLUE-中文医疗大模型评测基准](https://github.com/michael-wzhu/PromptCBLUE)



## Updates

2023/07/23 provide examples of TS NLI tasks; 19 tasks completed

2023/06/27 some of the task will be formulated as time series NLI tasks, due to lack of meaningful label descriptions; 

2023/06/26 processing UCR time-series classification archieve; 



## 免责声明

- 本项目相关资源仅供学术研究之用，严禁用于商业用途。
- Logo中的小学霸羊驼,与[Chinese-LlaMA2大模型](https://github.com/michael-wzhu/Chinese-LlaMA2), [PromptCBLUE](https://github.com/michael-wzhu/PromptCBLUE), [ChatMed](https://github.com/michael-wzhu/ChatMed)项目共用logo, 是由[midjourney](http://midjourney.com)自动生成的。


## 技术交流

PromptCBLUE与大模型技术交流微信交流群二维码（截止至7月23日有效）：
<p align="left">
    <br>
    <img src="./pics/wechat_qrcode.jpg" width="300"/>
    <br>
</p>


## 团队介绍

本项目由华东师范大学计算机科学与技术学院智能知识管理与服务团队完成，团队指导老师为王晓玲教授。

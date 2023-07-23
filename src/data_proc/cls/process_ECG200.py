import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
The following time series traces the electrical activity recorded during one heartbeat: <ts-data-blank-sep>. Data points are seperated by blank spaces.
Could you determine whether this time series corresponds to a normal heartbeat or a myocardial infarction event? 
<explanation>
a myocardial infarction event means a heart attack due to prolonged cardiac ischemia
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The given time series represents the electrical activity recorded during a single heartbeat: <ts-data-blank-sep>. Each data point is separated by blank spaces.
Can you please analyze the time series to identify if it corresponds to a normal heartbeat or indicates a myocardial infarction event?
<explanation>
A myocardial infarction event, commonly known as a heart attack, occurs when there is prolonged cardiac ischemia.
<response>
""",
    """<query>
I have a dataset that contains the daily stock prices of a company over the past year. Could you help me analyze the data and identify any significant trends or patterns?
<explanation>
Analyzing the stock price data can provide insights into the performance and behavior of the company's stock over time.
<response>
""",
    """<query>
The provided time series displays the electrical activity recorded during a single heartbeat: <ts-data-blank-sep>. Each data point is separated by blank spaces.
Could you please determine whether this time series indicates a normal heartbeat or signifies a myocardial infarction event?
<explanation>
A myocardial infarction event, also known as a heart attack, is caused by prolonged cardiac ischemia.
<response>
""",
    """<query>
The time series given below represents the electrical activity recorded during one heartbeat: <ts-data-blank-sep>. The data points are separated by blank spaces.
Can you analyze this time series to ascertain if it corresponds to a normal heartbeat or signifies a myocardial infarction event?
<explanation>
A myocardial infarction event refers to a heart attack caused by prolonged cardiac ischemia.
<response>
""",
    """<query>
Please evaluate the following time series that captures the electrical activity recorded during a single heartbeat: <ts-data-blank-sep>. Each data point is separated by blank spaces.
Is it possible to determine whether this time series corresponds to a normal heartbeat or indicates a myocardial infarction event?
<explanation>
A myocardial infarction event, also known as a heart attack, occurs due to prolonged cardiac ischemia.
<response>
""",
    """<query>
The given time series provides a trace of the electrical activity recorded during one heartbeat: <ts-data-blank-sep>. Each data point is separated by blank spaces.
Can you determine if this time series represents a normal heartbeat or signifies a myocardial infarction event?
<explanation>
A myocardial infarction event refers to a heart attack caused by prolonged cardiac ischemia.
<response>
""",
    """<query>
Examine the following time series that depicts the electrical activity recorded during a single heartbeat: <ts-data-blank-sep>. Data points are separated by blank spaces.
Please identify whether this time series corresponds to a normal heartbeat or indicates a myocardial infarction event.
<explanation>
A myocardial infarction event, commonly known as a heart attack, occurs due to prolonged cardiac ischemia.
<response>
""",
    """<query>
The following time series traces the electrical activity recorded during one heartbeat: <ts-data-comma-sep>. Data points are seperated by commas.
Could you determine whether this time series corresponds to a normal heartbeat or a myocardial infarction event? 
<explanation>
a myocardial infarction event means a heart attack due to prolonged cardiac ischemia
<response>
""",
    """<query>
The provided time series represents the electrical activity recorded during a single heartbeat: <ts-data-comma-sep>. Each data point is separated by commas.
Can you please analyze the time series to determine if it corresponds to a normal heartbeat or indicates a myocardial infarction event?
<explanation>
A myocardial infarction event, commonly known as a heart attack, occurs due to prolonged cardiac ischemia.
<response>
""",
    """<query>
Provided below is a time series depicting the electrical activity recorded during a single heartbeat: <ts-data-comma-sep>. Data points are separated by commas.
Can you determine if this time series corresponds to a normal heartbeat or indicates a myocardial infarction event?
<explanation>
A myocardial infarction event, also known as a heart attack, occurs as a result of prolonged cardiac ischemia.
<response>
""",
    """<query>
The time series presented shows the electrical activity recorded during one heartbeat: <ts-data-comma-sep>. Data points are listed with commas as separators.
Is it possible to determine whether this time series represents a normal heartbeat or indicates a myocardial infarction event?
<explanation>
A myocardial infarction event refers to a heart attack caused by prolonged cardiac ischemia.
<response>
""",
    """<query>
下面的时间序列追踪了一次心跳期间记录的电活动：<ts-data-comma-sep>。数据点之间用逗号分隔。
你能确定这个时间序列是对应正常心跳还是心肌梗塞事件吗？
<explanation>
心肌梗塞事件意味着由于心肌缺血时间过长而引发心脏病发作。
<response>
""",
    """<query>
下面展示的时间序列显示了一次心跳期间记录的电活动：<ts-data-comma-sep>。数据点用逗号分隔。
您能判断这个时间序列是代表正常心跳还是心肌梗塞事件吗？
<explanation>
心肌梗塞事件指的是由于心肌缺血时间过长而引发的心脏病发作。
<response>
""",

]

label_map = {
    "1": "normal heartbeat",
    "-1": "myocardial infarction event",
}
label_map_zh = {
    "1": "正常心跳",
    "-1": "心肌梗塞事件",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "ECG200"
to_folder = os.path.join(
    "datasets/prompt_datasets", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"datasets/UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 96
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)

                # 确认数据怎么分割的
                if "comma-sep" in template_:
                    template_.replace(
                        "<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_.replace(
                        "<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                # 确认中文还是英文
                target = ""
                if "序列" in template_:
                    target = label_zh_name_
                else:
                    target = label_name_

                list_datas.append(
                    {
                        "original_data": (label_name_, ts_data_blank_sep),
                        "target": target,
                        "input": template_,
                        "task_name": task_name,
                        "task_type": "classification",
                    }
                )


    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )

import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面的时间序列记录了一次心跳中的心电图数据：<ts-data-blank-sep>。数据来源于一位患有严重充血性心力衰竭的患者。
您能判断这个时间序列描述的心跳活动是属于哪一类别的吗？
<options>
可能的类别有：正常心跳，R-on-T型室性早搏，室上性早搏或异位搏动，室性早搏，其他类型心跳。
请从上述选项中选择一个回答。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
以下时间序列记录了一名患有严重充血性心力衰竭的患者的心电图数据：<ts-data-blank-sep>。
现在，请您判断这个时间序列描述的心跳活动属于以下哪一类别？
<options>
可能的类别包括：正常心跳，R-on-T型室性早搏，室上性早搏或异位搏动，室性早搏，其他类型心跳。
请从以上选项中选择一个答案。
<response>
    """,
    """<query>
下面的时间序列记录了一位严重充血性心力衰竭患者的心电图数据：<ts-data-blank-sep>。
您能判断这个时间序列所描述的心跳活动属于哪一类别吗？
<options>
可选类别：正常心跳，R-on-T型室性早搏，室上性早搏或异位搏动，室性早搏，其他类型心跳。
请从以上选项中选择一个答案。
<response>
    """,
    """<query>
这里记录了一组时间序列，反映了一名患有严重充血性心力衰竭的患者的心电图数据：<ts-data-blank-sep>。
您能判定这组时间序列所描述的心跳活动属于哪种类别吗？
<options>
可供选择的类别包括：正常心跳，R-on-T型室性早搏，室上性早搏或异位搏动，室性早搏，其他类型心跳。
请从以上选项中作出选择。
<response>
    """,

    """<query>
The following time series represents an electrocardiogram (ECG) data recorded during a heartbeat from a patient suffering from severe congestive heart failure: <ts-data-blank-sep>.
Can you determine which category of heartbeat activity this time series describes?
<options>
Possible categories include: Normal heartbeat, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Other types of heartbeat.
Please select one response from the options provided.
<response>
    """,
    """<query>
The given time series records the electrocardiogram (ECG) data of a patient experiencing severe congestive heart failure during a heartbeat: <ts-data-blank-sep>.
Could you identify the specific category to which this heartbeat activity belongs?
<options>
Potential categories are: Normal heartbeat, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Other types of heartbeat.
Please choose a response from the provided options.
<response>
    """,
    """<query>
In the provided time series, you can find the electrocardiogram (ECG) data from a patient with severe congestive heart failure during a heartbeat: <ts-data-blank-sep>.
Can you categorize the type of heartbeat activity represented in this time series?
<options>
You may choose from the following categories: Normal heartbeat, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Other types of heartbeat.
Please select one option as your response.
<response>
    """,
    """<query>
This time series presents the electrocardiogram (ECG) data recorded during a heartbeat from a patient afflicted with severe congestive heart failure: <ts-data-blank-sep>.
Would you be able to identify the category to which this particular heartbeat activity belongs?
<options>
Possible categories include: Normal heartbeat, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Other types of heartbeat.
Kindly choose one response from the options provided.
<response>
    """,
    """<query>
The provided time series displays the electrocardiogram (ECG) data of a patient suffering from severe congestive heart failure during a heartbeat: <ts-data-blank-sep>.
Your task is to determine the specific category of heartbeat activity described in this time series.
<options>
Available categories are: Normal heartbeat, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Other types of heartbeat.
Please make your selection from the given options.
<response>
    """,


]

label_map = {
    "1": "Normal heartbeat",
    "2": "R-on-T premature ventricular contraction",
    "3": "Supraventricular premature or ectopic beat",
    "4": "Premature ventricular contraction",
    "5": "Other types of heartbeat",
}
label_map_zh = {
    "1": "正常心跳",
    "2": "R-on-T型室性早搏",
    "3": "室上性早搏或异位搏动",
    "4": "室性早搏",
    "5": "其他类型心跳",  
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "ECG5000"
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
            assert len(ts_data_) == 140
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)
                # print(template_)

                # 确认数据怎么分割的
                if "<ts-data-comma-sep>" in template_:
                    template_ = template_.replace(
                        r"<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_ = template_.replace(
                        r"<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                assert "<ts-data-comma-sep>" not in template_
                assert "<ts-data-blank-sep>" not in template_

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

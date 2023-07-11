import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面的时间序列记录了心电图数据：<ts-data-blank-sep>。数据点用空格分割。
您能判断这个时间序列描述的心跳活动是属于哪一类别的吗？
<options>
可能的类别有：正常心跳，R-on-T型室性早搏，室上型早搏，室性早搏，未分类心跳。
请从上述选项中选择一个回答。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The following time series records the ECG data: <ts-data-blank-sep>. Data points are separated by spaces.
Can you determine which category of heartbeat activity this time series describes?
<options>
Possible categories: Normal, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Unclassified beat.
Please select one answer from the options above.
<response>
""",
    """<query>
我们有一个时间序列数据集（用空格分隔）：<ts-data-blank-sep>。它记录了心电图数据。请您判断这些数据属于哪一类心跳活动？
<options>
可能的类别有：正常心跳，R-on-T型室性早搏，室上型早搏，室性早搏，未分类心跳。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
We have a time series dataset (separated by spaces): <ts-data-blank-sep>. It records ECG data. Could you please determine the category of heartbeat activity for this data?
<options>
Possible categories: Normal, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Unclassified beat.
Please select one answer from the options above.
<response>
""",
    """<query>
以下时间序列记录了心电图数据：<ts-data-blank-sep>，其中数据点以空格分隔。请问您能确定这个时间序列描述的心跳活动属于哪一个类别？
<options>
可能的类别有：正常心跳，R-on-T型室性早搏，室上型早搏，室性早搏，未分类心跳。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
The following time series records the ECG data: <ts-data-blank-sep>, with data points separated by spaces. Can you determine which category of heartbeat activity this time series describes?
<options>
Possible categories: Normal, R-on-T premature ventricular contraction, Supraventricular premature or ectopic beat, Premature ventricular contraction, Unclassified beat.
Please select one answer from the options above.
<response>
""",


]

label_map = {
    "1": "Normal",
    "2": "R-on-T premature ventricular contraction",
    "3": "Supraventricular premature or ectopic beat",
    "4": "Premature ventricular contraction",
    "5": "Unclassified beat",
}
label_map_zh = {
    "1": "正常心跳",
    "2": "R-on-T型室性早搏",
    "3": "室上型早搏",
    "4": "室性早搏",
    "5": "未分类心跳",
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

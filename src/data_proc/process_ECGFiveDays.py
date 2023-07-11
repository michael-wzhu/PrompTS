import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面的时间序列数据取自于一个67岁男性在1990/11/12和1990/11/17两个日期的心电图：<ts-data-blank-sep>。数据点用空格分割。
您能判断这个数据是属于哪一天的吗？
<options>
可能的日期有：1990/11/12，1990/11/17。
请从上述选项中选择一个回答。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The time series data provided is from an electrocardiogram (ECG) of a 67-year-old male taken on two different dates, November 12, 1990, and November 17, 1990. Can you determine which day this data belongs to?
<options>
Possible dates are: 1990/11/12, 1990/11/17.
Please select one of the options provided above.
<response>
""",
    """<query>
我们有一个时间序列数据集（用空格分隔）：<ts-data-blank-sep>。它是从一个67岁男性的心电图中提取的，数据采集于1990年11月12日和1990年11月17日两天。请您判断这些数据属于哪一天？
<options>
可能的日期是：1990/11/12，1990/11/17。
请从上述选项中选择一个答案。
<response>
""",
    """<query>
We have a time series dataset separated by spaces: <ts-data-blank-sep>. It was extracted from an electrocardiogram of a 67-year-old male, collected on November 12, 1990, and November 17, 1990. Can you determine which day these data belong to?
<options>
Possible dates are: 1990/11/12, 1990/11/17.
Please select one of the options provided above.
<response>
""",
    """<query>
以下时间序列是一个67岁男性的心电图：<ts-data-blank-sep>，其中数据点以空格分隔。
这个时间序列采集于1990年11月12日和1990年11月17日，请判断该数据是属于哪一天的？
<options>
可能的日期是：1990/11/12，1990/11/17。
请从上述选项中选择一个答案。
<response>
""",
    """<query>
The following time series is an electrocardiogram of a 67-year-old male: <ts-data-blank-sep>, where data points are separated by spaces.
This time series was collected on November 12, 1990, and November 17, 1990. Can you determine which day this data belongs to?
<options>
Possible dates are: 1990/11/12, 1990/11/17.
Please select one of the options provided above.
<response>
""",


]

label_map = {
    "1": "1990/11/12",
    "2": "1990/11/17",
}
label_map_zh = {
    "1": "1990/11/12",
    "2": "1990/11/17",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "ECGFiveDays"
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
            assert len(ts_data_) == 136
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

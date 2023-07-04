import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
By utilizing a food spectrometer, we categorize food types and have obtained triple lane data for ham represented in the following time series: <ts-data-blank-sep>. Could you aid in distinguishing whether this ham is Spanish ham or French dry-cured ham?
<response>
"""

list_templates_by_chatgpt = [
    """<query>
We classify food types using a food spectrometer. The following time series represents the triple lane data extracted for ham: <ts-data-blank-sep>. The data points are separated by spaces. Can you help discern whether this ham is Spanish ham or French dry-cured ham?
<response>
""",
    """<query>
Can you assist in determining whether this ham is Spanish ham or French dry-cured ham by analyzing the triple lane data extracted using a food spectrometer represented in the following time series: <ts-data-blank-sep>?
<response>
""",
    """<query>
我们使用食品光谱仪对食品类型进行分类，以下时间序列代表对火腿这种食品提取的三重泳道数据：<ts-data-blank-sep>。数据点之间使用空格分隔。请你帮忙分辨该火腿是西班牙火腿还是法国干腌火腿？
<response>
""",
    """<query>
We employ a food spectrometer to classify food types, and the triple lane data extracted for ham is represented by the following time series: <ts-data-blank-sep>. Can you please assist in determining if this ham is Spanish ham or French dry-cured ham?
<response>
""",
    """<query>
To classify food types, we utilize a food spectrometer. The sequential time series indicates the triple-lane data extracted specifically for ham: <ts-data-blank-sep>. Spaces are used to separate the data points. Can you please aid in differentiating whether this ham is Spanish ham or French dry-cured ham?
<response>
""",
    """<query>
We categorize the food types by employing a food spectrometer. The provided time series represents the triple-lane data extracted for ham: <ts-data-blank-sep>. The data points are separated by spaces. Can you assist in distinguishing if this ham is Spanish ham or French dry-cured ham?
<response>
""",
]

label_map = {
    "1": "Spanish hams",
    "2": "French dry-cured hams",
}
label_map_zh = {
    "1": "西班牙火腿",
    "2": "法国干腌火腿",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Ham"
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
            assert len(ts_data_) == 431
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

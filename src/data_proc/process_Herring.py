import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
We are examining a time series dataset (separated by blank spaces): <ts-data-blank-sep>, which contains the Otolith contour data of a certain type of herring. Our task was to determine whether the sequence belonged to North Sea herring or Thames River herring. Can you help us determine the specific category?
<explanation>
Ear stones are calcium carbonate structures that exist in the lower portion of the cochlea, present in numerous vertebrate animals.
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The following time series is derived from the Otolith contour data of a certain type of herring: <ts-data-comma-sep>. The data is separated by spaces. Please determine whether this time series corresponds to North Sea herring or Thames herring.
<explanation>
Carbonate structures composed of calcium, known as ear stones, can be found within the lower region of the vestibule in various vertebrate species.
<response>
""",
    """<query>
We are currently examining a time series dataset (separated by spaces): <ts-data-blank-sep>, that comprises the Otolith contour information of a particular herring species. Our assignment was to establish whether the sequence corresponds to North Sea herring or Thames River herring. Can you aid us in determining the exact category?
<explanation>
Otoliths, made of calcium carbonate, are present within the lower part of the vestibule in multiple vertebrate species.
<response>
""",
    """<query>
The dataset we are scrutinizing is a time series (separated by blank spaces): <ts-data-blank-sep>, which encompasses the Otolith contour data of a specific herring type. Our primary goal was to establish whether the sequence belonged to North Sea herring or Thames River herring. Could you assist us in pinpointing the specific category?
<explanation>
The lower part of the cochlea houses ear stones, which are calcium carbonate structures found in vertebrate animals.
<response>
""",
    """<query>
We are currently engaged in the examination of a time series dataset (separated by spaces): <ts-data-blank-sep>, which includes the Otolith contour data of a specific herring variant. Our objective was to determine whether the sequence belongs to North Sea herring or Thames River herring. Can you provide assistance in identifying the specific category?
<explanation>
Ear stones, composed of calcium carbonate, are located inside the lower sphere in numerous vertebrates.
<response>
""",
    """<query>
Can you help us in determining the specific category of the time series dataset <ts-data-blank-sep> which contains the Otolith contour data of a certain type of herring? We were examining it to determine whether the sequence belonged to North Sea herring or Thames River herring.
<explanation>
In many vertebrate creatures, there are calcium carbonate structures known as otoliths, which reside within the lower sphere.
<response>
""",
    """<query>
下面的时间序列来自于某类鲱鱼的Otholith轮廓数据：<ts-data-comma-sep>。数据之间使用空格间隔。请你判断一下这个时间序列是北海鲱鱼还是泰晤士河鲱鱼？
<explanation>
耳石是许多脊椎动物中存在的碳酸钙结构，存在于下部的球囊内。
<response>
""",
    """<query>
我们正在研究一个时间序列数据集（由空格分隔）<ts-data-blank-sep>，其中包含一种特定种类鲱鱼的耳石轮廓数据。我们的目标是确定这个序列是属于北海鲱鱼还是泰晤士河鲱鱼。你能帮我们确定具体的分类吗？
<explanation>
许多脊椎动物中的下部球囊内存在着由碳酸钙结构组成的耳石。
<response>
""",
    """<query>
我们正在检查一个时间序列数据集（由空格分隔），其中包含一种特定种类鲱鱼的耳石轮廓数据<ts-data-blank-sep>。我们的任务是确定这个序列是属于北海鲱鱼还是泰晤士河鲱鱼。你可以帮助我们确定具体类别吗？
<explanation>
耳石是存在于许多脊椎动物内部下部球囊中的碳酸钙结构。
<response>
""",
]

label_map = {
    "1": "North sea herring",
    "2": "Thames herring",
}
label_map_zh = {
    "1": "北海鲱鱼",
    "2": "泰晤士鲱鱼",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Herring"
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
            assert len(ts_data_) == 512
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

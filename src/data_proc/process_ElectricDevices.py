import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面的时间序列记录了消费者在家中的用电行为数据：<ts-data-blank-sep>。数据点用空格分割，每个序列的长度为720，在24小时内每两分钟记录一次。
您能判断这个时间序列是监测的哪一种电器类型吗？
<options>
可能的电器类型有：显示器组，洗碗机，冷藏组，浸入式加热器，烧水壶，烤箱灶具，洗衣机。
请从上述选项中选择一个回答。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The following time series records consumer's electricity usage behavior at home: <ts-data-blank-sep>. Data points are separated by spaces, and each sequence has a length of 720. The data is recorded every two minutes within a 24-hour period.
Can you determine which type of appliance this time series is monitoring?
<options>
Possible appliance types: screenGroup, dishwasher, coldGroup, immersionHeater, kettle, ovenCooker, washingMachine.
Please select one answer from the options above.
<response>
""",
    """<query>
我们提供了一个时间序列数据集（用空格分隔）：<ts-data-blank-sep>。它记录了消费者在家中的用电行为数据。每个序列长度为720，表示在24小时内每两分钟记录一次。请确定这个时间序列对应的是哪种电器类型。
<options>
可能的电器类型有：显示器组，洗碗机，冷藏组，浸入式加热器，烧水壶，烤箱灶具，洗衣机。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
We have provided a time series dataset (separated by spaces): <ts-data-blank-sep>. It records the consumer's electricity usage behavior at home. Each sequence has a length of 720, representing a recording every two minutes over a 24-hour period. Please determine which type of appliance this time series corresponds to.
<options>
Possible appliance types: screenGroup, dishwasher, coldGroup, immersionHeater, kettle, ovenCooker, washingMachine.
Please select one answer from the options above.
<response>
""",
    """<query>
以下是一个时间序列记录消费者在家中的用电行为数据：<ts-data-blank-sep>。每个序列长度为720，代表在24小时内每两分钟记录一次。请您确定这个时间序列对应的是哪一种电器类型。
<options>
可能的电器类型有：显示器组，洗碗机，冷藏组，浸入式加热器，烧水壶，烤箱灶具，洗衣机。
请从上述选项中选择一个回答。
<response>
""",
    """<query>
Here is a time series that records the electricity usage behavior of a consumer at home: <ts-data-blank-sep>. Each sequence has a length of 720, representing a recording every two minutes over a 24-hour period. Can you determine which type of appliance this time series corresponds to?
<options>
Possible appliance types: screenGroup, dishwasher, coldGroup, immersionHeater, kettle, ovenCooker, washingMachine.
Please select one answer from the options above.
<response>
""",


]

label_map = {
    "1": "screenGroup",
    "2": "dishwasher",
    "3": "coldGroup",
    "4": "immersionHeater",
    "5": "kettle",
    "6": "ovenCooker",
    "7": "washingMachine"
}
label_map_zh = {
    "1": "显示器组",
    "2": "洗碗机",
    "3": "冷藏组",
    "4": "浸入式加热器",
    "5": "烧水壶",
    "6": "烤箱灶具",
    "7": "洗衣机"
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "ElectricDevices"
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

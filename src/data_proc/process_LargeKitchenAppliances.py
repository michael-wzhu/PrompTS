import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
以下是收集到的消费者在家中用电量的时间序列数据：<ts-data-blank-sep>，其中数据点以空格分隔。请根据该时间序列判断消费者使用的电器是哪个。
<option>
从洗衣机、滚筒式烘干机和洗碗机中选择一个。
不可以回答其他答案。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
已知消费者在家中用电量的时间序列数据：<ts-data-blank-sep>，其中数据点以空格分隔。请根据该时间序列判断消费者使用的电器。
<option>
从洗衣机、滚筒式烘干机和洗碗机中选择一个。
不可以回答其他答案。
<response>
""",
    """<query>
给出消费者在家中用电量的时间序列数据：<ts-data-blank-sep>，其中数据点以空格分隔。根据该时间序列，请回答消费者使用的是哪种电器。
<option>
答案从洗衣机、滚筒式烘干机和洗碗机中选择一个。
不可以回答其他答案。
<response>
""",
    """<query>
Below is the time series data of electricity consumption by consumers in their homes: <ts-data-blank-sep>, where data points are separated by spaces. Please determine which appliance the consumer is using based on this time series.
<option>
Choose between a washing machine, a tumble dryer, or a dishwasher.
Do not provide any other answer.
<response>
""",
    """<query>
Presented is a time series dataset of electricity consumption by consumers in their homes: <ts-data-blank-sep>, with data points separated by spaces. Based on this time series, please identify the appliance being used by the consumer.
<option>
Choose between a washing machine, a tumble dryer, or a dishwasher.
Do not provide any other answer.
<response>
""",
    """<query>
Give the time series data of consumers' electricity consumption at home: <ts-data-blank-sep>, where the data points are separated by spaces. Based on this time series, answer what kind of appliance the consumer is using.
<option>
The answer is one of the washing machine, tumble dryer and dishwasher.
No other answer can be given.
<response>
""",

]

label_map = {
    "1": "washing machine",
    "2": "tumble dryer",
    "3": "dishwasher",
}
label_map_zh = {
    "1": "洗衣机",
    "2": "滚筒式烘干机",
    "3": "洗碗机",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "LargeKitchenAppliances"
to_folder = os.path.join(
    "E:\\硕\\Time_Series_Instruct-main", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"E:\\硕\\Time_Series_Instruct-main\\UCRArchive_2018\\{task_name}\\{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 720
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

import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
以下是消费者在家中使用某种家用电器时的24小时用电量时间序列数据（每两分钟记录一次）：<ts-data-blank-sep>。请根据该时间序列判断消费者使用的电器是哪个类型。
<option>
从洗衣机、滚筒式烘干机和洗碗机中选择一个。
不可以回答其他答案。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
以下是消费者在家中使用某种家用电器时的24小时用电量时间序列数据（每两分钟记录一次）：<ts-data-blank-sep>。请根据该时间序列判断消费者使用的电器是哪个类型。
<option>
请选择消费者使用的电器类型：洗衣机、滚筒式烘干机或洗碗机。
请勿提供其他答案。
<response>
    """,
    """<query>
以下是消费者在家中使用某种家用电器时的24小时用电量时间序列数据（每两分钟记录一次）：<ts-data-blank-sep>。请根据该时间序列分析消费者使用的电器类型。
<option>
请从以下选项中选择消费者使用的电器类型：
1. 洗衣机
2. 滚筒式烘干机
3. 洗碗机
<response>
    """,
    """<query>根据提供的数据，推断消费者在家中使用的家用电器类型是哪一个？\n<ts-data-blank-sep>, 这是消费者在家中使用某种家用电器时，每两分钟的用电量时间序列数据
<option>
请选择消费者使用的电器类型：
1. 洗衣机
2. 滚筒式烘干机
3. 洗碗机
<response>
    """,

    
    """<query>Based on the provided data, determine the type of household appliance that the consumer is using at home? \n<ts-data-blank-sep>, these are time series data representing the electricity consumption every two minutes when the consumer is using a certain household appliance at home.
<option>
Please select the type of household appliance the consumer is using:
1. washing machine
2. tumble dryer
3. dishwasher
<response>
    """,
    """<query>Based on the given data, make an inference about the household appliance type being used by the consumer at their place of residence? \n<ts-data-blank-sep>, these time series data indicate the electricity usage every two minutes while the consumer employs a specific household appliance in their home.
<option>
Kindly choose the type of household appliance the consumer is utilizing:
1. washing machine
2. tumble dryer
3. dishwasher
<response>
    """,
    """<query>
Below is the 24-hour electricity consumption time series data of a certain household appliance used by the consumer at home (recorded every two minutes): <ts-data-blank-sep>. Please determine which type of household appliance the consumer is using based on this time series.
<option>
Choose one from the washing machine, tumble dryer, and dishwasher.
Do not provide any other answers.
<response>
    """,
    """<query>
The following is a time series data of electricity consumption for a specific household appliance used by the consumer at home over a 24-hour period (recorded every two minutes): <ts-data-blank-sep>. Based on this time series, please identify the type of household appliance being used by the consumer.
<option>
Select one from the washing machine, tumble dryer, and dishwasher.
Do not include any other options.
<response>
    """,
    """<query>
Here is the 24-hour electricity usage time series data for a certain household appliance used by the consumer at home (recorded every two minutes): <ts-data-blank-sep>. Your task is to determine the type of household appliance the consumer is using based on this time series.
<option>
Choose one among the washing machine, tumble dryer, and dishwasher.
Please refrain from providing other answers.
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

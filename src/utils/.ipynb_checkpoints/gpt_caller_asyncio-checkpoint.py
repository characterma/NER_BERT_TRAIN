import logging
import asyncio, json, nest_asyncio, aiohttp

import pandas as pd

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm


logger = logging.getLogger("python_config_logger")
logging.basicConfig(
    handlers=[
        logging.FileHandler(filename="log.txt", mode="w+", encoding="utf-8"),
        logging.StreamHandler()
    ],
    level=logging.DEBUG,
    format = '%(asctime)s %(message)s',
    datefmt = '%Y-%m-%d %H:%M:%S'
)



# ========================== 异步调用 ==========================
N_JOB = 50
MIN_WAIT_RANDOM_EXP = 1
MAX_WAIT_RANDOM_EXP = 5
STOP_AFTER_ATTEMPT = 2

class AsyncioRequestGPTCaller:
    nest_asyncio.apply()  # for jupyter issue.

    def __init__(
        self,
        TEMPLATE_ID: int,
        tags: str,
        njobs: int = N_JOB,
        API_URL: str = "http://aiapi.wisers.com/openai-result-service-api/invoke",
    ):
        self.TEMPLATE_ID = TEMPLATE_ID
        self.API_URL = API_URL
        self.tags = tags
        self.njobs = njobs

    async def _send_request(self, session, doc_id, template_data):
        async with session.post(self.API_URL, json=template_data) as resp:
            data = await resp.json()
            # print(f"请求数据：{template_data}: 响应结果：{data}")
            if data["response_json"] is not None:
                return {"doc_id": doc_id, "llm_response": data["response_json"]}
            elif data["response_text"] is not None:
                return {"doc_id": doc_id, "llm_response": data["response_text"]}
            else:
                logger.error("response result is except")

    @retry(
        wait=wait_random_exponential(min=MIN_WAIT_RANDOM_EXP, max=MAX_WAIT_RANDOM_EXP),
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    )
    async def _bound_request(self, sem, session, doc_id, template_data):
        try:
            async with sem:
                # print(template_data)
                return await self._send_request(session, doc_id, template_data)
        except Exception as error:
            logger.error(f"error: {error}, {template_data}")
            # raise error
            return {"doc_id": doc_id, "llm_response": None, "error": error}

    async def _async_get_response(self, doc_id_list, data_list, progress=True):
        tasks = []
        sem = asyncio.Semaphore(self.njobs)
        async with aiohttp.ClientSession() as session:
            for doc_id, data in zip(doc_id_list, data_list):
                task = asyncio.ensure_future(
                    self._bound_request(sem, session, doc_id, data)
                )
                tasks.append(task)

            data_list = []
            for f in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), disable=not (progress)
            ):
                body = await f
                data_list.append(body)
            return data_list

    def _request_post_format_transform(
        self, data, system_message_variable: list = [], **kwargs 
    ): #system_message_variable: json = {},
        if isinstance(data, pd.DataFrame):
            # check input format
            for col in ["headline", "content"]:
                assert col in data.columns, ValueError(data.columns)

            doc_id_list, data_list = [], []
            for i, (index, value) in enumerate(data.iterrows()):
                doc_id = value.get("doc_id")
                template_data = {
                    "template_id": self.TEMPLATE_ID,
                    "tags": [self.tags],
                    "data": value.to_dict(),
                }
                template_data.update(**kwargs)
                if doc_id is None:
                    template_data["data"]["doc_id"] = index
                if len(system_message_variable) > 0:
                    template_data["data"]["prompt_params_override"] = system_message_variable[i] ## mike改

                doc_id_list.append(doc_id if doc_id is not None else index)
                data_list.append(template_data)
            return doc_id_list, data_list

        elif isinstance(data, list):
            # check input format
            if len(data):
                for col in ["headline", "content"]:
                    assert col in data[0].keys(), ValueError(list(data[0].keys()))

            doc_id_list, data_list = [], []
            for index, value in enumerate(data):
                doc_id = value.get("doc_id")
                template_data = {
                    "template_id": self.TEMPLATE_ID,
                    "tags": [self.tags],
                    "data": value,
                }
                template_data.update(**kwargs)
                if doc_id is None:
                    template_data["data"]["doc_id"] = index
                if len(system_message_variable) > 0:
                    template_data["data"]["prompt_params_override"] = system_message_variable[index] ## mike改

                doc_id_list.append(doc_id if doc_id is not None else index)
                data_list.append(template_data)
            return doc_id_list, data_list

        else:
            raise TypeError(type(data))

    def run(self, input_data, progress=False, **kwargs):
        """Call this function to start the process"""
        doc_id_list, data_list = self._request_post_format_transform(input_data, **kwargs)

        # print(data_list)

        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(
            self._async_get_response(doc_id_list, data_list, progress)
        )
        responses = loop.run_until_complete(future)
        return responses


def get_gpt_result(input_data, template_code, api_url, semaphore_num, tags, **kwargs):
    """ 
    Input: 
        input_data: list of dicts, e.g. {"doc_id" : item['key'], "headline": "", "content": item['content']}
    """
    caller = AsyncioRequestGPTCaller(
        TEMPLATE_ID=template_code, tags=tags, API_URL=api_url, njobs=semaphore_num
    )

    output_data = caller.run(input_data=input_data, progress=True, **kwargs)
    
    output = {}
    for item in output_data:
        try:
            output.update({item['doc_id']: item.get("llm_response")})
        except Exception as e:
            print(e)
            # output.update({item['doc_id']: None})
    return output


if __name__ == "__main__":

    gpt_input = [
        {
            "doc_id": item['subaspect_desp'],   ##改
            "headline": '',                     ##改
            "content": item['subaspect_desp']   ##改
        } 
        for _, item in df_vkg.iterrows()
    ]

    industry_list = [{"industry": i} for i in ...]  ##改

    gpt_output = get_gpt_result(
        input_data = gpt_input, 
        template_code = 234,  ##改
        api_url = "http://aiapi.wisers.com/openai-result-service-api/invoke", 
        semaphore_num = 60, 
        tags = "",  ##改
        system_message_variable = industry_list
    )

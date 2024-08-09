from typing import Dict

from graphrag.query.structured_search.base import BaseSearch
from plugins.webserver import prompt
from plugins.webserver.service.graphrag.universal import build_search_engine
from plugins.webserver.types import GraphRAGItem, SourceEnum


class AsyncSearchEngineContext:
    def __init__(self, search_engine_args: Dict, request: GraphRAGItem):
        self.search_engine_args: dict = search_engine_args
        self.request: GraphRAGItem = request
        self.other_search_kwargs = {}

    def set_system_prompt(self):
        if self.request.source == SourceEnum.qa:
            system_prompt = prompt.LOCAL_SEARCH_FOR_QA_SYSTEM_PROMPT
        else:  # chat
            system_prompt = prompt.LOCAL_SEARCH_FOR_CHAT_SYSTEM_PROMPT
        return system_prompt

    def set_llm_params(self):
        if self.request.response_max_token:
            llm_params = {'max_tokens': self.request.response_max_token}
        else:
            llm_params = {}

        return llm_params

    def set_context_builder_params(self):
        if self.request.context_max_token:
            context_builder_params = {'max_tokens': self.request.context_max_token}
        else:
            context_builder_params = {}
        return context_builder_params

    def update_search_engine_args(self, **kwargs):
        # update search engine args
        self.search_engine_args.update(**kwargs)

    async def __aenter__(self):
        self.other_search_kwargs['add_history_to_search_messages'] = True \
            if self.request.source == SourceEnum.chat else False
        system_prompt = self.set_system_prompt()  # set system prompt
        llm_params = self.set_llm_params()  # set llm params
        context_builder_params = self.set_context_builder_params()  # set context builder params

        new_search_engine_args = {
            'system_prompt': system_prompt,
            'llm_params': llm_params,
            'context_builder_params': context_builder_params,
            'response_type': self.request.response_type
        }

        self.update_search_engine_args(**new_search_engine_args)  # update search engine args

        # instance a search engine
        search_engine: BaseSearch = build_search_engine(
            mode=self.request.method,
            **self.search_engine_args,
        )
        self.search_engine = search_engine

        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

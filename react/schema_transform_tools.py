from typing import Any, Dict
from langchain.prompts import PromptTemplate
from prompts import *
import llm_models

class SchemaTransformTools:
    def __init__(self, source_schema, target_schema, source_examples=None, target_examples=None):
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.source_examples = source_examples
        self.target_examples = target_examples

    def prompt_constructor(self, template, include_vars=None, additional_vars=None):
        vars_to_include = {}
        for var in include_vars:
            if hasattr(self, var):
                vars_to_include[var] = getattr(self, var)
            else:
                raise AttributeError(f"The attribute '{var}' is not found in the instance.")
        if additional_vars is not None:
            vars_to_include.update(additional_vars)
        final_prompt = template.format(**vars_to_include)
        return final_prompt

    def result_extractor(self, full_response):
        try:
            result: str = full_response.split('[START]')[1].split('[END]')[0]
            return result
        except Exception as e:
            print(f"Error extracting result: {e}")
            return None

    def call_llm(self, model_version, template, include_vars, additional_vars=None):
        try:
            final_prompt = self.prompt_constructor(template, include_vars, additional_vars)
            if model_version == 'gpt3.5':
                full_response = llm_models.gpt3(final_prompt)
            else:
                full_response = llm_models.gpt4(final_prompt)
            return self.result_extractor(full_response)
        except Exception as e:
            print(f"An error occurred while calling the language model: {e}")
            return None

    def type_predict(self) -> str:
        return self.call_llm('gpt4', type_predict_template, ["source_schema", "target_schema",
                                                             "source_examples", "target_examples"])

    def column_mapping(self) -> str:
        return self.call_llm('gpt4', column_mapping_template, ["source_schema", "target_schema"])

    def aggregation(self) -> str:
        return self.call_llm('gpt4', aggregation_template, ["source_schema", "target_schema",
                                                            "source_examples", "target_examples"])

    def clarify(self, question) -> str:
        result = input(question)
        return result

    def conditional(self) -> str:
        return self.call_llm('gpt4', conditional_template, ["source_schema", "target_schema"])

    def finish(self, prompt) -> str:
        additional_vars = {
            "prompt": prompt
        }
        result = self.call_llm('gpt4', finish_template, include_vars=[],
                             additional_vars=additional_vars)#.replace('\n', ' ') # this will corrupt the SQL comments

        if '```sql' in result:
            result = result.split('```sql')[1].split('```')[0]
        return result

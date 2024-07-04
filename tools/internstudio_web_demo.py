
# isort: skip_file
import copy
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

logger = logging.get_logger(__name__)
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

import argparse

import os
import random
current_token_len = 0
def longest_dup_substring(s: str) -> str:
    # 生成两个进制
    a1, a2 = random.randint(26, 100), random.randint(26, 100)
    # 生成两个模
    mod1, mod2 = random.randint(10**9+7, 2**31-1), random.randint(10**9+7, 2**31-1)
    n = len(s)
    # 先对所有字符进行编码
    arr = [ord(c)-ord('a') for c in s]
    # 二分查找的范围是[1, n-1]
    l, r = 1, n-1
    length, start = 0, -1
    while l <= r:
        m = l + (r - l + 1) // 2
        idx = check(arr, m, a1, a2, mod1, mod2)
        # 有重复子串，移动左边界
        if idx != -1:
            l = m + 1
            length = m
            start = idx
        # 无重复子串，移动右边界
        else:
            r = m - 1
    return s[start:start+length] if start != -1 else ""

def check(arr, m, a1, a2, mod1, mod2):
    n = len(arr)
    aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
    h1, h2 = 0, 0
    for i in range(m):
        h1 = (h1 * a1 + arr[i]) % mod1
        h2 = (h2 * a2 + arr[i]) % mod2
    # 存储一个编码组合是否出现过
    seen = {(h1, h2)}
    for start in range(1, n - m + 1):
        h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
        h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
        # 如果重复，则返回重复串的起点
        if (h1, h2) in seen:
            return start
        seen.add((h1, h2))
    # 没有重复，则返回-1
    return -1

class ForbidDuplicationProcessor(LogitsProcessor):
    """
    防止生成的内容陷入循环。
    当循环内容与循环次数之乘积大于指定次数
    则在生成下一个token时将循环内容的第一个token概率设置为0
    ---------------
    ver: 2023-08-17
    by: changhongyu
    """
    def __init__(self, tokenizer, threshold: int = 10):
        self.tokenizer = tokenizer
        self.threshold = threshold
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        global current_sequence_len
        current_sequence = self.tokenizer.decode(input_ids[0][current_token_len: ])
        current_dup_str = longest_dup_substring(current_sequence)
        if len(current_dup_str):
            # 如果存在重复子序列，则根据其长度与重复次数判断是否禁止循环
            if len(current_dup_str) > 1 or (len(current_dup_str) == 1 and current_dup_str * self.threshold in current_sequence):
                if len(current_dup_str) * current_sequence.count(current_dup_str) >= self.threshold:
                    token_ids = self.tokenizer.encode(current_dup_str)
                    # 获取截止目前的上一个token
                    last_token = input_ids[0][-1].detach().cpu().numpy().tolist()
                    if len(token_ids) and last_token == token_ids[-1]:
                        # 如果截止目前的上一个token，与重复部分的最后一个token一致
                        # 说明即将触发重复, 先把重复部分的第一个token禁掉
                        scores[:, token_ids[0]] = 0
                        # 然后按出现比率判断是否重复部分内还有其他重复
                        for token_id in token_ids:
                            if token_ids.count(token_id) * len(token_ids) > 1.2:
                                scores[:, token_id] = 0

        return scores

class MaxConsecutiveProcessor(LogitsProcessor):
    """
    给定一个集合，集合中的字符最多连续若干次
    下一次生成时不能再出现该集合中的字符
    ---------------
    ver: 2023-08-17
    by: changhongyu
    ---------------
    修复bug
    ver: 2023-09-11
    """
    def __init__(self, consecutive_token_ids, max_num: int = 10):
        self.consecutive_token_ids = consecutive_token_ids
        self.max_num = max_num
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        input_ids_list = input_ids.squeeze(0).detach().cpu().numpy().tolist()
        cur_num = 0
        for token in input_ids_list[::-1]:
            if token in self.consecutive_token_ids:
                cur_num += 1
            else:
                break
                
        if cur_num >= self.max_num:
            # 如果连续次数超过阈值，那集合中的所有token在下一个step都不可以再出现
            for token_id in self.consecutive_token_ids:
                scores[..., token_id] = 0
        return scores

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 32768
    top_p: float = 0.8
    temperature: float = 0.8
    do_sample: bool = True
    repetition_penalty: float = 1.005


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model(arg1):
    # model = AutoModelForCausalLM.from_pretrained(args.m).cuda()
    # tokenizer = AutoTokenizer.from_pretrained(args.m, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(arg1, torch_dtype=torch.float16,trust_remote_code=True).cuda()
    tokenizer = AutoTokenizer.from_pretrained(arg1, trust_remote_code=True)

  
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=8192,
                               value=8192)
        top_p = st.slider('Top P', 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.7, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>'
robot_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n{robot}<|eot_id|>'
cur_query_prompt = '<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'


def combine_history(prompt):
    messages = st.session_state.messages
    total_prompt = ''
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main(arg1):
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model(arg1)
    print('load model end.')

    st.title('Llama3 DoctorAssist')

    generation_config = prepare_generation_config()

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('Hello!'):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        ###qwtl

        embeddings=HuggingFaceEmbeddings(model_name="/home/yzc/llamawork/text2vec-large-chinese", model_kwargs={'device': 'cuda'})
        #如果之前没有本地的faiss仓库，就把doc读取到向量库后，再把向量库保存到本地
        if os.path.exists("/home/yzc/llamawork/my_faiss_store.faiss")==False:
            vector_store=FAISS.from_documents(docs,embeddings)
            vector_store.save_local("/home/yzc/llamawork/my_faiss_store.faiss")
        #如果本地已经有faiss仓库了，说明之前已经保存过了，就直接读取
        else:
            vector_store=FAISS.load_local("/home/yzc/llamawork/my_faiss_store.faiss",embeddings,allow_dangerous_deserialization=True)

        
        docs=vector_store.similarity_search_with_score(prompt)#计算相似度，并把相似度高的chunk放在前面
        context = []
        for i in docs:
            print(i[1])
            if i[1]<521:
                context.append(i[0].page_content)
        print(context)
        system_prompt = """
        <<SYS>>
        你是一个医疗人工智能助手。
        你的名字是LlamaDoctor
        <</SYS>>

        """
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })
        if(len(context) > 0 ): prompt = "请回答问题：" +  prompt + '\n' +  "以下是上下文信息如下。：\n" + str(context)
        ###
        real_prompt = system_prompt + combine_history(prompt)
        # Add user message to chat history
        
        logits_processor = LogitsProcessorList()
        number_tokens = [str(i) for i in range(10)] + ['.', '-']
        number_token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in number_tokens]
        logits_processor.append(ForbidDuplicationProcessor(tokenizer))
        logits_processor.append(MaxConsecutiveProcessor(number_token_ids))
        with st.chat_message('robot'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(

                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    logits_processor=logits_processor,
                    additional_eos_token_id=128009,  # <|eot_id|>
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':

    import sys
    arg1 = sys.argv[1]
    main(arg1)

package com.cc.personal.llmapidemo.constant;

public class Constant {
    /** 用的是小米的LLM模型的apikey **/
    public static final String LLM_API_KEY =  "tp-c5hnu1zqjqmmw4p1gvamrfdr7bgx6ylt78npeejop8x4ema0";
    public static final String LLM_MODEL_NAME = "mimo-v2.5-pro";
    /** 其他模型通过阿里云百炼调用 */
    public static final String ALIBAILIAN_API_KEY =  "sk-31f62dc869f14c1686c2e933acd65aa4";

    // LLM url
    public static final String LLM_API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions";

    //embedding url
    public static final String EMBEDDING_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings";
    // 重排序url
    public static final String RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks";



}

package com.cc.personal.llmapidemo.constant;

public class Constant {
    /** 用的是小米的LLM模型的apikey **/
    public static final String LLM_API_KEY =  "tp-c5hnu1zqjqmmw4p1gvamrfdr7bgx6ylt78npeejop8x4ema0";
    public static final String LLM_MODEL_NAME = "mimo-v2.5-pro";
    /** 其他模型通过阿里云百炼调用 */
    public static final String ALIBAILIAN_API_KEY =  "sk-31f62dc869f14c1686c2e933acd65aa4";

    // LLM url
    public static final String LLM_API_URL = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions";

    // 小LLM url 通过alibailian调用
    public static final String Q_LLM_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
    public static final String Q_MODEL_NAME = "qwen2.5-7b-instruct";


    //embedding url
    public static final String EMBEDDING_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings";
    // 重排序url
    public static final String RERANK_URL = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks";
    // 小chat模型的modelname 通过阿里调用的千问一个2.5B的模型 可以用于指令重写等地方


}

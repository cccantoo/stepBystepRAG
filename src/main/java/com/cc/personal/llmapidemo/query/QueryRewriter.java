package com.cc.personal.llmapidemo.query;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.OkHttpClient;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.List;
import okhttp3.*;

import static com.cc.personal.llmapidemo.constant.Constant.*;

/**
 *用mimo的大模型总是容易出现超时 但是用千问的一个2.5B的小模型出结果很快 且tokens消耗少
 *
 * 查询改写助手，需求
 * 1. 用户可能问题中的代词不明确指代，应该根据上下文分析改写该问题再拿给LLM询问
 * 2. 上下文补全
 * 3. 口语化转正式
 *
 * 生产环境的注意事项：
 * 1. 改写质量的监控   改写率/过度改写率/改写后检索提升率
 * 2. 改写失败的兜底   若请求Query改写失败，用原始 query 检索也能有一定的效果。不能改写失败就让整个 RAG 流程挂掉。
 */
public class QueryRewriter {

    private static final String API_URL = Q_LLM_API_URL; // 这里其实可以用便宜的小模型 因为只是改写query的请求
    private static final String API_KEY = ALIBAILIAN_API_KEY;
    // 改写用小模型就够了，成本低、速度快
    private static final String MODEL = Q_MODEL_NAME;
    private static final OkHttpClient client = new OkHttpClient();
    private static final Gson gson = new Gson();

    /**
     * 改写 Prompt 模板
     */
    private static final String REWRITE_PROMPT = """
            你是一个查询改写助手。根据对话历史和用户的最新问题，\
            将问题改写为一个独立的、完整的检索查询。
            
            要求：
            1. 如果最新问题中包含代词（它、这个、那个等）或省略了关键信息，\
            请结合对话历史补全
            2. 如果问题已经足够完整清晰，请原样输出，不要画蛇添足
            3. 不要添加用户没有提到的信息
            4. 只输出改写后的查询，不要输出任何解释、前缀或多余内容
            5. 改写后的查询应该是一个独立的句子，脱离对话历史也能理解
            
            对话历史：
            %s
            
            用户最新问题：%s
            
            改写后的查询：""";

    /**
     * 执行 Query 改写
     *
     * @param history      对话历史（role + content 的列表）
     * @param currentQuery 用户当前问题
     * @return 改写后的查询
     */
    public static String rewrite(List<Message> history,
                                 String currentQuery) throws IOException {
        // 构建对话历史文本
        StringBuilder historyText = new StringBuilder();
        if (history.isEmpty()) {
            historyText.append("（无历史对话）");
        } else {
            for (Message msg : history) {
                String roleName = "user".equals(msg.role) ? "用户" : "助手";
                historyText.append(roleName).append("：")
                        .append(msg.content).append("\n");
            }
        }

        // 构建改写请求
        JsonObject body = getJsonObject(currentQuery, historyText);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(body.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            assert response.body() != null;
            String responseBody = response.body().string();
            JsonObject json = gson.fromJson(responseBody, JsonObject.class);
            System.out.println(json);
            return json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString().trim();
        }
    }

    private static JsonObject getJsonObject(String currentQuery, StringBuilder historyText) {
        String prompt = String.format(REWRITE_PROMPT,
                historyText.toString(), currentQuery);

        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("temperature", 0.1);
        body.addProperty("max_tokens", 1024);
        JsonArray messages = new JsonArray();
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", prompt);
        messages.add(userMsg);
        body.add("messages", messages);
        return body;
    }

    /**
     * 简单的消息数据结构
     */
    public static class Message {
        public String role;
        public String content;

        public Message(String role, String content) {
            this.role = role;
            this.content = content;
        }
    }

    public static void main(String[] args) throws IOException {
        // ===== 场景 1：指代消解 =====
        System.out.println("===== 场景 1：指代消解 =====");
        List<Message> history1 = List.of(
                new Message("user", "iPhone 16 Pro 的退货政策是什么？"),
                new Message("assistant",
                        "iPhone 16 Pro 因屏幕定制工艺，拆封后不支持七天无理由退货。" +
                                "如有质量问题，可联系售后处理。")
        );
        String rewritten1 = rewrite(history1, "那它的保修期呢？");
        System.out.println("原始 query：那它的保修期呢？");
        System.out.println("改写结果：" + rewritten1);

        // ===== 场景 2：上下文补全 =====
        System.out.println("\n===== 场景 2：上下文补全 =====");
        List<Message> history2 = List.of(
                new Message("user", "iPhone 16 Pro 有什么颜色？"),
                new Message("assistant",
                        "iPhone 16 Pro 有沙漠色钛金属、自然色钛金属、" +
                                "白色钛金属和黑色钛金属四种颜色。")
        );
        String rewritten2 = rewrite(history2, "价格呢？");
        System.out.println("原始 query：价格呢？");
        System.out.println("改写结果：" + rewritten2);

        // ===== 场景 3：口语化转正式 =====
        System.out.println("\n===== 场景 3：口语化转正式 =====");
        List<Message> history3 = List.of();  // 无历史，第一轮对话
        String rewritten3 = rewrite(history3, "东西坏了咋整？");
        System.out.println("原始 query：东西坏了咋整？");
        System.out.println("改写结果：" + rewritten3);

        // ===== 场景 4：已经完整的 query，不需要改写 =====
        System.out.println("\n===== 场景 4：不需要改写 =====");
        List<Message> history4 = List.of();
        String rewritten4 = rewrite(history4,
                "iPhone 16 Pro 的退货政策是什么？");
        System.out.println("原始 query：iPhone 16 Pro 的退货政策是什么？");
        System.out.println("改写结果：" + rewritten4);
    }
}

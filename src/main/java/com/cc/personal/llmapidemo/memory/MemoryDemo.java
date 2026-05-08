package com.cc.personal.llmapidemo.memory;

import com.google.gson.*;
import okhttp3.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static com.cc.personal.llmapidemo.constant.Constant.*;

/**
 * 比较发送请求询问大模型的时候 携带记忆和无记忆的对比
 */
public class MemoryDemo {

    private static final String API_URL = LLM_API_URL;
    private static final String API_KEY = LLM_API_KEY;
    private static final String MODEL = LLM_MODEL_NAME;
//    private static final OkHttpClient client = new OkHttpClient(); 默认的超时是10s容易超时断开
    private static final int MAX_RETRIES = 3;
    private static final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .writeTimeout(30,TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build();
    private static final Gson gson = new Gson();

    public static void main(String[] args) throws IOException {
//        System.out.println("===== 无记忆模式 =====");
//        noMemoryDemo();

//        System.out.println("\n===== 有记忆模式 =====");
//        withMemoryDemo();

//        System.out.println("\n===== 滑动窗口保留记忆模式 =====");
//        slidingWindowMemoryDemo();

        System.out.println("\n===== 摘要压缩保留记忆模式 =====");
        compressMemoryDemo();


    }


    // 摘要压缩记忆保留
    public static void compressMemoryDemo() throws IOException {
        SummaryMemory memory = new SummaryMemory(600,2); // 达到多少token就触发摘要压缩，前期的压缩，近期的2轮保留
        String sessionId = "session-002";
        String systemPrompt = "你是一个电商客服助手，简洁回答用户问题。";

        // 模拟 6 轮对话
        String[] questions = {
                "iPhone 15 Pro 多少钱？",          // 第 1 轮
                "有什么颜色可选？",                  // 第 2 轮
                "白色的有现货吗？",                  // 第 3 轮
                "支持分期吗？",                     // 第 4 轮（第 1 轮被丢弃）
                "那它的退货政策呢？"   ,              // 第 5 轮（第 2 轮被丢弃）
                "再说一下我一开始询问的手机的价格呢？"
        };

        for (String question : questions) {
            List<JsonObject> messages = memory.buildMessages(
                    sessionId, systemPrompt, question);
            // 调用 API
            String answer = chat(messages);
            // 保存对话历史
            memory.addMessage(sessionId, "user", question);
            memory.addMessage(sessionId, "assistant", answer);

            System.out.println("用户：" + question);
            System.out.println("助手：" + answer);
            System.out.println();
        }
    }





    // 滑动窗口记忆保留
    public static void slidingWindowMemoryDemo() throws IOException {
        // 创建一个只保留最近 3 轮对话的记忆管理器
        SlidingWindowMemory memory = new SlidingWindowMemory(3);
        String sessionId = "session-001";
        String systemPrompt = "你是一个电商客服助手，简洁回答用户问题。";

        // 模拟 5 轮对话
        String[] questions = {
                "iPhone 16 Pro 多少钱？",          // 第 1 轮
                "有什么颜色可选？",                  // 第 2 轮
                "白色的有现货吗？",                  // 第 3 轮
                "支持分期吗？",                     // 第 4 轮（第 1 轮被丢弃）
                "那它的退货政策呢？"   ,              // 第 5 轮（第 2 轮被丢弃）
                "再说一下我想要的手机的价格呢？"
        };

        for (String question : questions) {
            // 构建消息（自动应用滑动窗口）
            List<JsonObject> messages = memory.buildMessages(
                    sessionId, systemPrompt, question);
            // 调用 API
            String answer = chat(messages);
            // 保存对话历史
            memory.addMessage(sessionId, "user", question);
            memory.addMessage(sessionId, "assistant", answer);

            System.out.println("用户：" + question);
            System.out.println("助手：" + answer);
            System.out.println("当前记忆中的消息数：" +
                    memory.getRecentMessages(sessionId).size());
            System.out.println();
        }
    }


    /**
     * 无记忆模式：每次请求只带当前问题，不带历史消息
     */
    static void noMemoryDemo() throws IOException {
        // 第 1 轮
        String answer1 = chat(List.of(
                message("system", "你是一个电商客服助手，简洁回答用户问题。"),
                message("user", "iPhone 16 Pro 的退货政策是什么？")
        ));
        System.out.println("用户：iPhone 16 Pro 的退货政策是什么？");
        System.out.println("助手：" + answer1);

        // 第 2 轮：不带历史消息，模型不知道"它"是什么
        String answer2 = chat(List.of(
                message("system", "你是一个电商客服助手，简洁回答用户问题。并结合上下文信息回答。"), // 怎么都不会记忆 必须是携带上history
                message("user", "那它的保修期呢？")
        ));
        System.out.println("\n用户：那它的保修期呢？");
        System.out.println("助手：" + answer2);
    }

    /**
     * 有记忆模式：每次请求带上完整的历史消息
     */
    static void withMemoryDemo() throws IOException {
        // history作为记忆 每次会从history中取出信息全部发送到chat模型
        List<JsonObject> history = new ArrayList<>();
        history.add(message("system", "你是一个电商客服助手，简洁回答用户问题。"));

        // 第 1 轮
        history.add(message("user", "iPhone 16 Pro 的退货政策是什么？"));
        String answer1 = chat(history);
        history.add(message("assistant", answer1));
        System.out.println("用户：iPhone 16 Pro 的退货政策是什么？");
        System.out.println("助手：" + answer1);

        // 第 2 轮：带上第 1 轮的历史，模型知道"它"指 iPhone 16 Pro
        history.add(message("user", "那它的保修期呢？"));
        String answer2 = chat(history);
        history.add(message("assistant", answer2));
        System.out.println("\n用户：那它的保修期呢？");
        System.out.println("助手：" + answer2);

        // 第 3 轮：继续追问
        history.add(message("user", "过了保修期维修大概多少钱？"));
        String answer3 = chat(history);
        System.out.println("\n用户：过了保修期维修大概多少钱？");
        System.out.println("助手：" + answer3);
    }

    /**
     * 调用 mimo Chat API
     */
    static String chat(List<JsonObject> messages) throws IOException {
        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("temperature", 0.1);
        body.addProperty("max_tokens", 512);
        JsonArray messagesArray = new JsonArray();
        for (JsonObject msg : messages) {
            messagesArray.add(msg);
        }
        body.add("messages", messagesArray);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(body.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body().string();
            JsonObject json = gson.fromJson(responseBody, JsonObject.class);
            return json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        }
    }

    static JsonObject message(String role, String content) {
        JsonObject msg = new JsonObject();
        msg.addProperty("role", role);
        msg.addProperty("content", content);
        return msg;
    }
}

package com.cc.personal.llmapidemo.hybrid;

import cn.hutool.core.collection.CollUtil;
import com.cc.personal.llmapidemo.constant.Constant;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.milvus.common.clientenum.FunctionType;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.ConsistencyLevel;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq.Function;
import io.milvus.v2.service.collection.request.HasCollectionReq;
import io.milvus.v2.service.collection.request.LoadCollectionReq;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.vector.request.AnnSearchReq;
import io.milvus.v2.service.vector.request.HybridSearchReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.EmbeddedText;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.request.ranker.RRFRanker;
import io.milvus.v2.service.vector.response.InsertResp;
import io.milvus.v2.service.vector.response.SearchResp;
import lombok.SneakyThrows;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.*;

/**
 * Milvus 混合检索演示程序
 *
 * 功能：演示三种检索模式的对比
 * 1. DENSE_ONLY - 纯向量检索（稠密向量，使用余弦相似度）
 * 2. SPARSE_ONLY - 纯BM25检索（稀疏向量，基于词频和逆文档频率）
 * 3. HYBRID - Dense + Sparse混合检索（RRF融合排序）
 *
 * 数据流：
 * 用户查询文本 → Embedding API转换为向量 → Milvus检索 → 返回TopK结果
 */
public class MilvusHybridSchemaDemo {

    /** Collection名称，类似于数据库中的表名 */
    private static final String COLLECTION = "customer_service_hybrid";

    /** 阿里云DashScope API Key，用于调用Embedding模型 */
    private static final String ALIYUN_API_KEY = Constant.ALIBAILIAN_API_KEY;

    /** Embedding API地址（OpenAI兼容格式） */
    private static final String EMBEDDING_URL = Constant.EMBEDDING_API_URL;

    /** 使用的Embedding模型名称，输出1024维向量 */
    private static final String EMBEDDING_MODEL = "text-embedding-v3";

    /** Gson用于JSON序列化/反序列化 */
    private static final Gson GSON = new Gson();

    /** OkHttpClient用于调用Embedding API */
    private static final OkHttpClient HTTP_CLIENT = new OkHttpClient();

    /**
     * 三种检索模式枚举
     */
    public enum SearchMode {
        DENSE_ONLY,     // 纯向量检索：只使用稠密向量（FloatVector）
        SPARSE_ONLY,    // 纯 BM25 检索：只使用稀疏向量（SparseFloatVector）
        HYBRID          // Dense + Sparse 混合检索：同时使用两种向量，RRF融合排序
    }

    /**
     * 检索参数配置类
     * 包含各种检索参数，用于控制召回数量、性能和质量
     */
    public static class SearchConfig {
        /** Dense检索召回的候选数量（用于混合检索时） */
        public int denseRecallTopK = 20;

        /** Sparse检索召回的候选数量（用于混合检索时） */
        public int sparseRecallTopK = 20;

        /** 最终返回的TopK结果数量 */
        public int finalTopK = 8;

        /** 向量检索时的probe数量，影响召回率和性能，值越大召回率越高但性能越差 */
        public int nprobe = 16;

        /** BM25检索时的丢弃比例，用于性能优化，丢弃相似度最低的部分结果 */
        public double dropRatioSearch = 0.2;

        /** RRF（Reciprocal Rank Fusion）融合参数，值越大，排名靠后的结果权重越高：
         * rrfK 是平滑参数，控制不同排名之间的分数差异程度。
         * 相当于是原始rank/rrf rrf是除数 所以对于排名靠后的更友好
         *  rrfK=1 时：第1名分数是第10名的 5.5倍（0.500/0.091）
         * - rrfK=60 时：第1名分数是第10名的 1.15倍（0.0164/0.0143）
         * - rrfK=200 时：第1名分数是第10名的 1.05倍（0.00498/0.00476）*/
        public int rrfK = 60;

        /** 返回的字段列表（除了向量字段外） */
        public List<String> outFields = List.of("text");

        /** 一致性级别：BOUNDED表示读取最近一段时间内的数据，性能较好 */
        public ConsistencyLevel consistencyLevel = ConsistencyLevel.BOUNDED;

        /**
         * 静态工厂方法，返回默认配置
         * @return 默认的SearchConfig实例
         */
        public static SearchConfig defaults() {
            return new SearchConfig();
        }
    }

    /**
     * 主方法：程序入口
     * 1. 创建Milvus客户端连接
     * 2. 创建Collection（如果不存在）并加载到内存
     * 3. 依次运行三种检索模式进行对比
     */
    public static void main(String[] args) {
        // 创建Milvus客户端连接
        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://192.168.1.102:19530")
                .build());

        // 创建Collection（如果不存在）并加载到内存
        createCollectionIfAbsentAndLoad(client);

        // 用户查询文本
        String query = "订单号 2026012345 的物流状态";

        // 使用默认配置
        SearchConfig cfg = SearchConfig.defaults();

        // 依次运行三种检索模式进行对比
        for (SearchMode mode : SearchMode.values()) {
            SearchResp resp = runSearch(client, query, mode, cfg);
            printSearchResults(resp, mode);
        }
    }

    // ==================== Collection 创建与数据加载 ====================

    /**
     * 创建Collection（如果不存在）并加载到内存
     * 包含以下步骤：
     * 1. 定义Schema（表结构）
     * 2. 创建Collection
     * 3. 创建索引
     * 4. 插入示例数据
     * 5. 加载Collection到内存
     *
     * @param client Milvus客户端实例
     */
    public static void createCollectionIfAbsentAndLoad(MilvusClientV2 client) {
        // 检查Collection是否已存在
        Boolean exists = client.hasCollection(
                HasCollectionReq.builder().collectionName(COLLECTION).build()
        );

        if (!Boolean.TRUE.equals(exists)) {
            // ===== Collection不存在，需要创建 =====

            // 1) 定义Schema（表结构）
            CreateCollectionReq.CollectionSchema schema = client.createSchema();

            // 主键字段：Int64类型，自动递增
            schema.addField(AddFieldReq.builder()
                    .fieldName("id").dataType(DataType.Int64)
                    .isPrimaryKey(true).autoID(true).build());

            // 文本字段：VarChar类型，最大8192字符，启用分析器（用于BM25分词）
            schema.addField(AddFieldReq.builder()
                    .fieldName("text").dataType(DataType.VarChar)
                    .maxLength(8192).enableAnalyzer(true).build());

            // 稠密向量字段：FloatVector类型，1024维（text-embedding-v3输出维度）
            schema.addField(AddFieldReq.builder()
                    .fieldName("text_dense").dataType(DataType.FloatVector)
                    .dimension(1024).build());

            // 稀疏向量字段：SparseFloatVector类型（用于BM25检索）
            schema.addField(AddFieldReq.builder()
                    .fieldName("text_sparse").dataType(DataType.SparseFloatVector).build());

            // 添加BM25函数：自动从text字段生成稀疏向量
            // 当插入数据时，Milvus会自动计算text字段的BM25稀疏向量并存入text_sparse字段
            schema.addFunction(Function.builder()
                    .functionType(FunctionType.BM25)
                    .name("text_bm25_emb")           // 函数名称
                    .inputFieldNames(List.of("text"))  // 输入字段
                    .outputFieldNames(List.of("text_sparse"))  // 输出字段
                    .build());

            // 2) 创建Collection
            client.createCollection(CreateCollectionReq.builder()
                    .collectionName(COLLECTION).collectionSchema(schema).build());

            // 3) 创建索引
            // Dense向量索引：AUTOINDEX自动选择最优索引类型，使用余弦相似度（COSINE）
            IndexParam denseIndex = IndexParam.builder()
                    .fieldName("text_dense")
                    .indexType(IndexParam.IndexType.AUTOINDEX)
                    .metricType(IndexParam.MetricType.COSINE).build();

            // Sparse向量索引：AUTOINDEX，使用BM25相似度
            IndexParam sparseIndex = IndexParam.builder()
                    .fieldName("text_sparse")
                    .indexType(IndexParam.IndexType.AUTOINDEX)
                    .metricType(IndexParam.MetricType.BM25).build();

            // 创建索引
            client.createIndex(CreateIndexReq.builder()
                    .collectionName(COLLECTION)
                    .indexParams(List.of(denseIndex, sparseIndex)).build());

            // 4) 插入示例数据
            // 每条数据包含text字段和对应的dense向量
            // 注意：sparse向量由BM25函数自动生成，不需要手动插入
            List<JsonObject> rows = Arrays.asList(
                    buildRow("订单号 2026012345 的物流状态：已发货，预计 1 月 28 日送达，承运商顺丰速运。"),
                    buildRow("物流规则总述：标准订单 48 小时内发货，偏远地区可能延迟 1-2 天。"),
                    buildRow("发货时效说明：付款成功后，普通商品 24-48 小时内发货，预售商品以详情页为准。"),
                    buildRow("异常签收处理：如包裹显示已签收但未收到，请在 48 小时内联系客服核实。"),
                    buildRow("订单查询入口：登录 APP → 我的订单 → 输入订单号即可查看物流详情。"),
                    buildRow("退货政策：收到商品 7 天内可申请无理由退货，需保持商品完好。")
            );

            // 执行插入
            InsertResp insertResp = client.insert(InsertReq.builder()
                    .collectionName(COLLECTION).data(rows).build());
            System.out.println("插入数据条数：" + insertResp.getInsertCnt());
        }

        // 将Collection加载到内存（必须加载后才能检索）
        client.loadCollection(LoadCollectionReq.builder()
                .collectionName(COLLECTION).build());
        System.out.println("Collection 已就绪并加载：" + COLLECTION);
    }

    // ==================== 三种检索模式 ====================

    /**
     * 根据检索模式分发到不同的检索方法
     *
     * @param client Milvus客户端实例
     * @param queryText 用户查询文本
     * @param mode 检索模式（DENSE_ONLY/SPARSE_ONLY/HYBRID）
     * @param cfg 检索配置参数
     * @return 检索结果
     */
    @SneakyThrows
    public static SearchResp runSearch(MilvusClientV2 client,
                                       String queryText,
                                       SearchMode mode,
                                       SearchConfig cfg) {
        return switch (mode) {
            case DENSE_ONLY -> runDenseOnly(client, queryText, cfg);    // 纯向量检索
            case SPARSE_ONLY -> runSparseOnly(client, queryText, cfg);  // 纯BM25检索
            default -> runHybrid(client, queryText, cfg);               // 混合检索
        };
    }

    /**
     * 纯向量检索：只使用稠密向量（Dense Vector）
     * 原理：将查询文本转换为1024维浮点向量，使用余弦相似度计算与Collection中向量的距离
     *
     * @param client Milvus客户端实例
     * @param queryText 用户查询文本
     * @param cfg 检索配置参数
     * @return 检索结果
     */
    private static SearchResp runDenseOnly(MilvusClientV2 client,
                                           String queryText,
                                           SearchConfig cfg) throws IOException {
        // 1. 将查询文本转换为向量（调用阿里云Embedding API）
        List<Float> queryVec = getEmbedding(queryText);

        // 2. 设置检索参数
        Map<String, Object> params = new HashMap<>();
        params.put("metric_type", "COSINE");  // 使用余弦相似度
        params.put("nprobe", cfg.nprobe);      // probe数量，影响召回率和性能

        // 3. 执行向量检索 todo client.search
        return client.search(SearchReq.builder()
                .collectionName(COLLECTION)
                .annsField("text_dense")                              // 检索dense向量字段
                .data(Collections.singletonList(new FloatVec(queryVec)))  // 查询向量
                .topK(cfg.finalTopK)                                  // 返回TopK结果
                .outputFields(cfg.outFields)                          // 返回的字段
                .searchParams(params)                                 // 检索参数
                .consistencyLevel(cfg.consistencyLevel)               // 一致性级别
                .build());
    }

    /**
     * 纯BM25检索：只使用稀疏向量（Sparse Vector）
     * 原理：基于BM25算法计算查询文本与Collection中文本的相似度
     * BM25是经典的文本检索算法，基于词频（TF）和逆文档频率（IDF）计算相关性
     *
     * @param client Milvus客户端实例
     * @param queryText 用户查询文本
     * @param cfg 检索配置参数
     * @return 检索结果
     */
    private static SearchResp runSparseOnly(MilvusClientV2 client,
                                            String queryText,
                                            SearchConfig cfg) {
        // 1. 设置BM25检索参数
        Map<String, Object> params = new HashMap<>();
        params.put("metric_type", "BM25");                    // 使用BM25相似度
        params.put("drop_ratio_search", cfg.dropRatioSearch); // 丢弃比例，用于性能优化

        // 2. 执行BM25检索
        // 注意：使用EmbeddedText而不是FloatVec，Milvus会自动将文本转换为稀疏向量
        return client.search(SearchReq.builder()
                .collectionName(COLLECTION)
                .annsField("text_sparse")                                // 检索sparse向量字段
                .data(Collections.singletonList(new EmbeddedText(queryText)))  // 查询文本
                .topK(cfg.finalTopK)                                     // 返回TopK结果
                .outputFields(cfg.outFields)                             // 返回的字段
                .searchParams(params)                                    // 检索参数
                .consistencyLevel(cfg.consistencyLevel)                  // 一致性级别
                .build());
    }

    /**
     * 混合检索：Dense + Sparse，RRF融合排序
     * 原理：
     * 1. 同时执行Dense向量检索和Sparse BM25检索
     * 2. 使用RRF（Reciprocal Rank Fusion）算法融合两个检索结果
     * 3. RRF公式：score = Σ 1 / (k + rank_i)，其中k是平滑参数，rank_i是排名
     * 4. 混合检索结合了语义相似度（Dense）和关键词匹配（Sparse）的优势
     *
     * @param client Milvus客户端实例
     * @param queryText 用户查询文本
     * @param cfg 检索配置参数
     * @return 检索结果
     */
    private static SearchResp runHybrid(MilvusClientV2 client,
                                        String queryText,
                                        SearchConfig cfg) throws IOException {
        // 1. 将查询文本转换为向量（用于Dense检索）
        List<Float> queryVec = getEmbedding(queryText);

        // 2. 构建Dense检索请求
        AnnSearchReq denseReq = AnnSearchReq.builder()
                .vectorFieldName("text_dense")                              // 检索dense向量字段
                .vectors(Collections.singletonList(new FloatVec(queryVec))) // 查询向量
                .params("{\"nprobe\": " + cfg.nprobe + "}")                 // 检索参数
                .topK(cfg.denseRecallTopK)                                  // Dense召回TopK
                .build();

        // 3. 构建Sparse检索请求（BM25）
        AnnSearchReq sparseReq = AnnSearchReq.builder()
                .vectorFieldName("text_sparse")                                 // 检索sparse向量字段
                .vectors(Collections.singletonList(new EmbeddedText(queryText))) // 查询文本
                .params("{\"drop_ratio_search\": " + cfg.dropRatioSearch + "}")  // 检索参数
                .topK(cfg.sparseRecallTopK)                                     // Sparse召回TopK
                .build();

        // 4. 构建混合检索请求
        HybridSearchReq hybridReq = HybridSearchReq.builder()
                .collectionName(COLLECTION)
                .searchRequests(List.of(denseReq, sparseReq))  // 两个检索请求
                .ranker(new RRFRanker(cfg.rrfK))                // RRF融合排序器
                .topK(cfg.finalTopK)                            // 最终返回TopK
                .consistencyLevel(cfg.consistencyLevel)         // 一致性级别
                .outFields(cfg.outFields)                       // 返回字段
                .build();

        // 5. 执行混合检索
        return client.hybridSearch(hybridReq);
    }

    /**
     * 打印检索结果
     *
     * @param resp 检索响应对象
     * @param mode 检索模式
     */
    private static void printSearchResults(SearchResp resp, SearchMode mode) {
        System.out.println("\n===== Mode: " + mode + " =====");

        // 获取检索结果（支持多查询，这里只用了单查询）
        List<List<SearchResp.SearchResult>> results = resp.getSearchResults();

        for (List<SearchResp.SearchResult> oneQueryResults : results) {
            for (int i = 0; i < oneQueryResults.size(); i++) {
                SearchResp.SearchResult r = oneQueryResults.get(i);

                // 打印排名、相似度分数、ID
                System.out.println("Top-" + (i + 1) + " score=" + r.getScore() + ", id=" + r.getId());

                // 获取并打印text字段内容
                Object text = r.getEntity() == null ? null : r.getEntity().get("text");
                System.out.println("  " + text);
            }
        }
    }

    // ==================== 工具方法 ====================

    /**
     * 构建数据行：包含text字段和对应的dense向量
     * 用于插入示例数据到Collection
     *
     * @param text 文本内容
     * @return 包含text和text_dense字段的JsonObject
     */
    @SneakyThrows
    private static JsonObject buildRow(String text) {
        JsonObject row = new JsonObject();
        row.addProperty("text", text);  // 添加text字段

        // 获取text的embedding向量
        List<Float> denseVector = getEmbedding(text);

        // 将向量转换为JsonArray
        JsonArray arr = new JsonArray();
        for (Float f : denseVector) arr.add(f);
        row.add("text_dense", arr);  // 添加dense向量字段

        return row;
    }

    /**
     * 调用阿里云Embedding API生成稠密向量
     * API格式：OpenAI兼容格式
     * 模型：text-embedding-v3，输出1024维向量
     *
     * @param text 要转换的文本
     * @return 1024维浮点向量
     * @throws IOException API调用失败时抛出异常
     */
    private static List<Float> getEmbedding(String text) throws IOException {
        // 1. 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", EMBEDDING_MODEL);  // 模型名称
        requestBody.add("input", GSON.toJsonTree(List.of(text)));  // 输入文本（列表格式）

        // 2. 构建HTTP请求
        Request request = new Request.Builder()
                .url(EMBEDDING_URL)
                .addHeader("Authorization", "Bearer " + ALIYUN_API_KEY)  // API Key认证
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        GSON.toJson(requestBody),
                        MediaType.parse("application/json")))
                .build();

        // 3. 发送请求并处理响应
        try (Response response = HTTP_CLIENT.newCall(request).execute()) {
            String body = Objects.requireNonNull(response.body()).string();

            // 检查HTTP状态码
            if (!response.isSuccessful()) {
                throw new IOException("Embedding API 调用失败 http=" + response.code() + ", body=" + body);
            }

            // 4. 解析响应JSON
            JsonObject json = GSON.fromJson(body, JsonObject.class);
            JsonArray dataArray = json.getAsJsonArray("data");

            // 检查data数组是否为空
            if (CollUtil.isEmpty(dataArray)) {
                throw new IOException("Embedding API 返回 data 为空，原始响应: " + body);
            }

            // 5. 提取embedding向量
            JsonArray embeddingArray = dataArray.get(0).getAsJsonObject().getAsJsonArray("embedding");

            // 检查embedding是否为空
            if (embeddingArray == null) {
                throw new IOException("Embedding API 返回 embedding 为空，原始响应: " + body);
            }

            // 6. 转换为List<Float>
            List<Float> vector = new ArrayList<>(embeddingArray.size());
            for (int i = 0; i < embeddingArray.size(); i++) {
                vector.add(embeddingArray.get(i).getAsFloat());
            }
            return vector;
        }
    }
}

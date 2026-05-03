package com.cc.personal.llmapidemo.milvusdemo;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;

import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.index.request.CreateIndexReq;

import java.util.List;
import java.util.Map;

public class MilvusDemo {

    // 向量维度，和 Embedding 模型保持一致（text-Embedding-3 输出 1024 维）
    private static final int VECTOR_DIM = 1024;
    private static final String COLLECTION_NAME = "customer_service_chunks";

    public static void main(String[] args) {
        // 1. 连接 Milvus
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri("http://192.168.1.102:19530")
                .build();
        MilvusClientV2 client = new MilvusClientV2(connectConfig);
        System.out.println("已连接到 Milvus");

//        // 2. 定义 Schema
//        CreateCollectionReq.CollectionSchema schema = client.createSchema();
//
//        // 主键字段：自增 ID
//        schema.addField(AddFieldReq.builder()
//                .fieldName("id")
//                .dataType(DataType.Int64)
//                .isPrimaryKey(true)
//                .autoID(true)
//                .build());
//
//        // 向量字段：存储 Embedding 向量
//        schema.addField(AddFieldReq.builder()
//                .fieldName("vector")
//                .dataType(DataType.FloatVector)
//                .dimension(VECTOR_DIM)
//                .build());
//
//        // 标量字段：chunk 原文
//        schema.addField(AddFieldReq.builder()
//                .fieldName("chunk_text")
//                .dataType(DataType.VarChar)
//                .maxLength(8192)
//                .build());
//
//        // 标量字段：文档 ID（标识这个 chunk 来自哪个文档）
//        schema.addField(AddFieldReq.builder()
//                .fieldName("doc_id")
//                .dataType(DataType.VarChar)
//                .maxLength(64)
//                .build());
//
//        // 标量字段：分类（退货政策、物流规则、促销活动等）
//        schema.addField(AddFieldReq.builder()
//                .fieldName("category")
//                .dataType(DataType.VarChar)
//                .maxLength(32)
//                .build());
//
//        // 3. 创建 Collection
//        CreateCollectionReq createCollectionReq = CreateCollectionReq.builder()
//                .collectionName(COLLECTION_NAME)
//                .collectionSchema(schema)
//                .build();
//        client.createCollection(createCollectionReq);
//        System.out.println("Collection 创建成功：" + COLLECTION_NAME);


        // 为向量字段创建 HNSW 索引
        IndexParam vectorIndex = IndexParam.builder()
                .fieldName("vector")
                .indexType(IndexParam.IndexType.HNSW)
                .metricType(IndexParam.MetricType.COSINE)  // 余弦相似度
                .extraParams(Map.of(
                        "M", 16,              // 每个向量的最大连接数
                        "efConstruction", 256 // 建索引时的搜索宽度
                ))
                .build();

        // 为 category 标量字段创建索引（加速过滤查询）
        IndexParam categoryIndex = IndexParam.builder()
                .fieldName("category")
                .indexType(IndexParam.IndexType.TRIE)  // 字符串类型用 Trie 索引
                .build();

        CreateIndexReq createIndexReq = CreateIndexReq.builder()
                .collectionName("customer_service_chunks")
                .indexParams(List.of(vectorIndex, categoryIndex))
                .build();
        client.createIndex(createIndexReq);
        System.out.println("索引创建成功");
    }
}

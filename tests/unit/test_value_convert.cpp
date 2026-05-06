// =============================================================================
// test_value_convert.cpp — Testes de round-trip proto ↔ client::Value
//
// Cobre to_proto_value, from_proto_value, to_proto_struct, from_proto_struct.
// Requer enable_worker_tests=true (depende de protobuf linkado pelo worker_lib).
// =============================================================================

#include <gtest/gtest.h>
#include <client/value_convert.hpp>
#include <cmath>
#include <string>

using namespace mlinference::client;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static Value roundtrip(const Value& v) {
    return from_proto_value(to_proto_value(v));
}

static Object roundtrip_struct(const Object& o) {
    return from_proto_struct(to_proto_struct(o));
}

// =============================================================================
// Grupo 1: Null
// =============================================================================

TEST(ValueConvertNull, NullRoundtrip) {
    Value v;
    EXPECT_TRUE(roundtrip(v).is_null());
}

TEST(ValueConvertNull, NullToProtoHasNullKind) {
    auto pv = to_proto_value(Value{});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kNullValue);
}

// =============================================================================
// Grupo 2: Number (double)
// =============================================================================

TEST(ValueConvertNumber, PositiveRoundtrip) {
    EXPECT_DOUBLE_EQ(roundtrip(Value{3.14}).as_number(), 3.14);
}

TEST(ValueConvertNumber, NegativeRoundtrip) {
    EXPECT_DOUBLE_EQ(roundtrip(Value{-99.5}).as_number(), -99.5);
}

TEST(ValueConvertNumber, ZeroRoundtrip) {
    EXPECT_DOUBLE_EQ(roundtrip(Value{0.0}).as_number(), 0.0);
}

TEST(ValueConvertNumber, VeryLargeRoundtrip) {
    double big = 1e300;
    EXPECT_DOUBLE_EQ(roundtrip(Value{big}).as_number(), big);
}

TEST(ValueConvertNumber, VerySmallRoundtrip) {
    double tiny = 1e-300;
    EXPECT_DOUBLE_EQ(roundtrip(Value{tiny}).as_number(), tiny);
}

TEST(ValueConvertNumber, ProtoHasNumberKind) {
    auto pv = to_proto_value(Value{42.0});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kNumberValue);
    EXPECT_DOUBLE_EQ(pv.number_value(), 42.0);
}

// =============================================================================
// Grupo 3: Bool
// =============================================================================

TEST(ValueConvertBool, TrueRoundtrip) {
    EXPECT_EQ(roundtrip(Value{true}).as_bool(), true);
}

TEST(ValueConvertBool, FalseRoundtrip) {
    EXPECT_EQ(roundtrip(Value{false}).as_bool(), false);
}

TEST(ValueConvertBool, ProtoHasBoolKind) {
    auto pv = to_proto_value(Value{true});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kBoolValue);
    EXPECT_EQ(pv.bool_value(), true);
}

// =============================================================================
// Grupo 4: String
// =============================================================================

TEST(ValueConvertString, SimpleStringRoundtrip) {
    EXPECT_EQ(roundtrip(Value{std::string("hello")}).as_string(), "hello");
}

TEST(ValueConvertString, EmptyStringRoundtrip) {
    EXPECT_EQ(roundtrip(Value{std::string("")}).as_string(), "");
}

TEST(ValueConvertString, LongStringRoundtrip) {
    std::string s(5000, 'Z');
    EXPECT_EQ(roundtrip(Value{s}).as_string(), s);
}

TEST(ValueConvertString, SpecialCharsRoundtrip) {
    std::string s = "hello\nworld\t!@#$%";
    EXPECT_EQ(roundtrip(Value{s}).as_string(), s);
}

TEST(ValueConvertString, UnicodeRoundtrip) {
    std::string s = "日本語テスト";
    EXPECT_EQ(roundtrip(Value{s}).as_string(), s);
}

TEST(ValueConvertString, ProtoHasStringKind) {
    auto pv = to_proto_value(Value{std::string("test")});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kStringValue);
    EXPECT_EQ(pv.string_value(), "test");
}

// =============================================================================
// Grupo 5: Array
// =============================================================================

TEST(ValueConvertArray, EmptyArrayRoundtrip) {
    Value v{Array{}};
    auto rt = roundtrip(v);
    ASSERT_TRUE(rt.is_array());
    EXPECT_TRUE(rt.as_array().empty());
}

TEST(ValueConvertArray, NumberArrayRoundtrip) {
    Array arr{Value{1.0}, Value{2.0}, Value{3.0}};
    auto rt = roundtrip(Value{arr});
    ASSERT_TRUE(rt.is_array());
    ASSERT_EQ(rt.as_array().size(), 3u);
    EXPECT_DOUBLE_EQ(rt.as_array()[0].as_number(), 1.0);
    EXPECT_DOUBLE_EQ(rt.as_array()[1].as_number(), 2.0);
    EXPECT_DOUBLE_EQ(rt.as_array()[2].as_number(), 3.0);
}

TEST(ValueConvertArray, MixedTypeArrayRoundtrip) {
    Array arr{Value{42.0}, Value{true}, Value{std::string("x")}, Value{}};
    auto rt = roundtrip(Value{std::move(arr)});
    ASSERT_TRUE(rt.is_array());
    ASSERT_EQ(rt.as_array().size(), 4u);
    EXPECT_TRUE(rt.as_array()[0].is_number());
    EXPECT_TRUE(rt.as_array()[1].is_bool());
    EXPECT_TRUE(rt.as_array()[2].is_string());
    EXPECT_TRUE(rt.as_array()[3].is_null());
}

TEST(ValueConvertArray, LargeArrayRoundtrip) {
    Array arr;
    const int N = 1000;
    for (int i = 0; i < N; ++i)
        arr.push_back(Value{static_cast<double>(i)});
    auto rt = roundtrip(Value{std::move(arr)});
    ASSERT_TRUE(rt.is_array());
    ASSERT_EQ(rt.as_array().size(), static_cast<size_t>(N));
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(rt.as_array()[i].as_number(), static_cast<double>(i));
}

TEST(ValueConvertArray, ProtoHasListKind) {
    auto pv = to_proto_value(Value{Array{Value{1.0}, Value{2.0}}});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kListValue);
    EXPECT_EQ(pv.list_value().values_size(), 2);
}

// =============================================================================
// Grupo 6: Object
// =============================================================================

TEST(ValueConvertObject, EmptyObjectRoundtrip) {
    auto rt = roundtrip(Value{Object{}});
    ASSERT_TRUE(rt.is_object());
    EXPECT_TRUE(rt.as_object().empty());
}

TEST(ValueConvertObject, SimpleObjectRoundtrip) {
    Object obj;
    obj["a"] = Value{1.0};
    obj["b"] = Value{std::string("hello")};
    auto rt = roundtrip(Value{std::move(obj)});
    ASSERT_TRUE(rt.is_object());
    EXPECT_DOUBLE_EQ(rt.as_object().at("a").as_number(), 1.0);
    EXPECT_EQ(rt.as_object().at("b").as_string(), "hello");
}

TEST(ValueConvertObject, MultipleKeysRoundtrip) {
    Object obj;
    for (int i = 0; i < 20; ++i)
        obj["key_" + std::to_string(i)] = Value{static_cast<double>(i * 10)};
    auto rt = roundtrip(Value{std::move(obj)});
    ASSERT_TRUE(rt.is_object());
    EXPECT_EQ(rt.as_object().size(), 20u);
    for (int i = 0; i < 20; ++i)
        EXPECT_DOUBLE_EQ(rt.as_object().at("key_" + std::to_string(i)).as_number(),
                         static_cast<double>(i * 10));
}

TEST(ValueConvertObject, ProtoHasStructKind) {
    Object obj;
    obj["x"] = Value{5.0};
    auto pv = to_proto_value(Value{std::move(obj)});
    EXPECT_EQ(pv.kind_case(), google::protobuf::Value::kStructValue);
}

// =============================================================================
// Grupo 7: Aninhamento — Array dentro de Object
// =============================================================================

TEST(ValueConvertNested, ObjectWithArrayValues) {
    Object obj;
    obj["input"] = Value{Array{Value{1.0}, Value{2.0}, Value{3.0}}};
    obj["flags"]  = Value{Array{Value{true}, Value{false}}};
    auto rt = roundtrip(Value{std::move(obj)});
    ASSERT_TRUE(rt.is_object());
    EXPECT_EQ(rt.as_object().at("input").as_array().size(), 3u);
    EXPECT_EQ(rt.as_object().at("flags").as_array().size(), 2u);
    EXPECT_DOUBLE_EQ(rt.as_object().at("input").as_array()[1].as_number(), 2.0);
}

TEST(ValueConvertNested, ObjectInsideArray) {
    Array arr;
    Object inner;
    inner["val"] = Value{99.0};
    arr.push_back(Value{std::move(inner)});
    auto rt = roundtrip(Value{std::move(arr)});
    ASSERT_TRUE(rt.is_array());
    ASSERT_EQ(rt.as_array().size(), 1u);
    EXPECT_DOUBLE_EQ(rt.as_array()[0].as_object().at("val").as_number(), 99.0);
}

TEST(ValueConvertNested, DeepNesting) {
    Object outer;
    Object inner;
    Array innerArr{Value{42.0}};
    inner["arr"] = Value{std::move(innerArr)};
    outer["inner"] = Value{std::move(inner)};
    auto rt = roundtrip(Value{std::move(outer)});
    ASSERT_TRUE(rt.is_object());
    double v = rt.as_object().at("inner").as_object().at("arr").as_array()[0].as_number();
    EXPECT_DOUBLE_EQ(v, 42.0);
}

// =============================================================================
// Grupo 8: to_proto_struct / from_proto_struct
// =============================================================================

TEST(ValueConvertStruct, EmptyStructRoundtrip) {
    Object o;
    auto rt = roundtrip_struct(o);
    EXPECT_TRUE(rt.empty());
}

TEST(ValueConvertStruct, SingleKeyStructRoundtrip) {
    Object o;
    Array arr{Value{1.0}, Value{2.0}, Value{3.0}, Value{4.0}, Value{5.0}};
    o["input"] = Value{std::move(arr)};
    auto rt = roundtrip_struct(o);
    ASSERT_EQ(rt.size(), 1u);
    ASSERT_TRUE(rt.at("input").is_array());
    EXPECT_EQ(rt.at("input").as_array().size(), 5u);
}

TEST(ValueConvertStruct, MultipleKeysStructRoundtrip) {
    Object o;
    o["speed"]    = Value{12.5};
    o["active"]   = Value{true};
    o["mode"]     = Value{std::string("fast")};
    o["sensors"]  = Value{Array{Value{1.0}, Value{2.0}}};
    auto rt = roundtrip_struct(o);
    EXPECT_EQ(rt.size(), 4u);
    EXPECT_DOUBLE_EQ(rt.at("speed").as_number(), 12.5);
    EXPECT_EQ(rt.at("active").as_bool(), true);
    EXPECT_EQ(rt.at("mode").as_string(), "fast");
    EXPECT_EQ(rt.at("sensors").as_array().size(), 2u);
}

TEST(ValueConvertStruct, ToProtoStructHasCorrectFieldCount) {
    Object o;
    o["a"] = Value{1.0};
    o["b"] = Value{2.0};
    o["c"] = Value{3.0};
    auto ps = to_proto_struct(o);
    EXPECT_EQ(ps.fields_size(), 3);
}

// =============================================================================
// Grupo 9: Preservação de tipo após round-trip
// =============================================================================

TEST(ValueConvertTypePreservation, TypesPreservedInArray) {
    Array arr{
        Value{},                         // null
        Value{3.14},                     // number
        Value{true},                     // bool
        Value{std::string("hi")},        // string
        Value{Array{Value{1.0}}},        // array
        Value{Object{{"k", Value{2.0}}}} // object
    };
    auto rt = roundtrip(Value{std::move(arr)});
    const auto& a = rt.as_array();
    EXPECT_TRUE(a[0].is_null());
    EXPECT_TRUE(a[1].is_number());
    EXPECT_TRUE(a[2].is_bool());
    EXPECT_TRUE(a[3].is_string());
    EXPECT_TRUE(a[4].is_array());
    EXPECT_TRUE(a[5].is_object());
}

TEST(ValueConvertTypePreservation, NumberIsNotBool) {
    auto rt = roundtrip(Value{1.0});
    EXPECT_TRUE(rt.is_number());
    EXPECT_FALSE(rt.is_bool());
}

TEST(ValueConvertTypePreservation, BoolIsNotNumber) {
    auto rt = roundtrip(Value{true});
    EXPECT_TRUE(rt.is_bool());
    EXPECT_FALSE(rt.is_number());
}

// =============================================================================
// Grupo 10: Idempotência — aplicar round-trip duas vezes não muda o valor
// =============================================================================

TEST(ValueConvertIdempotent, DoubleRoundtripNumber) {
    Value v{123.456};
    auto rt1 = roundtrip(v);
    auto rt2 = roundtrip(rt1);
    EXPECT_DOUBLE_EQ(rt2.as_number(), 123.456);
}

TEST(ValueConvertIdempotent, DoubleRoundtripString) {
    Value v{std::string("idempotent")};
    auto rt1 = roundtrip(v);
    auto rt2 = roundtrip(rt1);
    EXPECT_EQ(rt2.as_string(), "idempotent");
}

TEST(ValueConvertIdempotent, DoubleRoundtripArray) {
    Value v{Array{Value{1.0}, Value{2.0}}};
    auto rt1 = roundtrip(v);
    auto rt2 = roundtrip(rt1);
    ASSERT_EQ(rt2.as_array().size(), 2u);
    EXPECT_DOUBLE_EQ(rt2.as_array()[0].as_number(), 1.0);
}
// =============================================================================
// test_value.cpp — Testes unitários do tipo Value / Object / Array
//
// Cobre construtores, checagem de tipo, acesso, cópia/move, aninhamento,
// edge cases numéricos e estruturas complexas.
// Sem dependência de servidor, modelo ou gRPC.
// =============================================================================

#include <gtest/gtest.h>
#include <client/inference_client.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <stdexcept>

using miia::client::Value;
using miia::client::Array;
using miia::client::Object;
using miia::client::Null;

// =============================================================================
// Grupo 1: Construtor default — Null
// =============================================================================

TEST(ValueDefault, IsNull) {
    Value v;
    EXPECT_TRUE(v.is_null());
}

TEST(ValueDefault, NotNumber) { EXPECT_FALSE(Value{}.is_number()); }
TEST(ValueDefault, NotBool)   { EXPECT_FALSE(Value{}.is_bool());   }
TEST(ValueDefault, NotString) { EXPECT_FALSE(Value{}.is_string()); }
TEST(ValueDefault, NotArray)  { EXPECT_FALSE(Value{}.is_array());  }
TEST(ValueDefault, NotObject) { EXPECT_FALSE(Value{}.is_object()); }

// =============================================================================
// Grupo 2: Construtor double
// =============================================================================

TEST(ValueDouble, IsNumber) {
    Value v{3.14};
    EXPECT_TRUE(v.is_number());
}

TEST(ValueDouble, ReturnsCorrectValue) {
    EXPECT_DOUBLE_EQ(Value{3.14}.as_number(), 3.14);
}

TEST(ValueDouble, Zero) {
    EXPECT_DOUBLE_EQ(Value{0.0}.as_number(), 0.0);
}

TEST(ValueDouble, Negative) {
    EXPECT_DOUBLE_EQ(Value{-1.5}.as_number(), -1.5);
}

TEST(ValueDouble, VeryLarge) {
    const double BIG = 1e308;
    EXPECT_DOUBLE_EQ(Value{BIG}.as_number(), BIG);
}

TEST(ValueDouble, VerySmall) {
    const double TINY = std::numeric_limits<double>::min();
    EXPECT_DOUBLE_EQ(Value{TINY}.as_number(), TINY);
}

TEST(ValueDouble, PositiveInfinity) {
    Value v{std::numeric_limits<double>::infinity()};
    EXPECT_TRUE(std::isinf(v.as_number()));
    EXPECT_GT(v.as_number(), 0.0);
}

TEST(ValueDouble, NegativeInfinity) {
    Value v{-std::numeric_limits<double>::infinity()};
    EXPECT_TRUE(std::isinf(v.as_number()));
    EXPECT_LT(v.as_number(), 0.0);
}

TEST(ValueDouble, NaN) {
    Value v{std::numeric_limits<double>::quiet_NaN()};
    EXPECT_TRUE(v.is_number());
    EXPECT_TRUE(std::isnan(v.as_number()));
}

// =============================================================================
// Grupo 3: Construtor float (promovido para double)
// =============================================================================

TEST(ValueFloat, IsNumber) {
    Value v{1.5f};
    EXPECT_TRUE(v.is_number());
}

TEST(ValueFloat, ValueIsPreservedAprox) {
    EXPECT_NEAR(Value{2.5f}.as_number(), 2.5, 1e-6);
}

// =============================================================================
// Grupo 4: Construtor int
// =============================================================================

TEST(ValueInt, IsNumber) {
    Value v{42};
    EXPECT_TRUE(v.is_number());
}

TEST(ValueInt, ReturnsCorrectValue) {
    EXPECT_DOUBLE_EQ(Value{42}.as_number(), 42.0);
}

TEST(ValueInt, Zero) {
    EXPECT_DOUBLE_EQ(Value{0}.as_number(), 0.0);
}

TEST(ValueInt, Negative) {
    EXPECT_DOUBLE_EQ(Value{-99}.as_number(), -99.0);
}

TEST(ValueInt, MaxInt) {
    EXPECT_DOUBLE_EQ(Value{2147483647}.as_number(),
                     static_cast<double>(2147483647));
}

// =============================================================================
// Grupo 5: Construtor int64_t
// =============================================================================

TEST(ValueInt64, IsNumber) {
    Value v{static_cast<int64_t>(1000000000LL)};
    EXPECT_TRUE(v.is_number());
}

TEST(ValueInt64, LargeValue) {
    int64_t big = 9000000000LL;
    EXPECT_DOUBLE_EQ(Value{big}.as_number(), static_cast<double>(big));
}

// =============================================================================
// Grupo 6: Construtor bool
// =============================================================================

TEST(ValueBool, IsBool) {
    EXPECT_TRUE(Value{true}.is_bool());
    EXPECT_TRUE(Value{false}.is_bool());
}

TEST(ValueBool, NotNumber) {
    EXPECT_FALSE(Value{true}.is_number());
}

TEST(ValueBool, TrueReturnsTrue) {
    EXPECT_EQ(Value{true}.as_bool(), true);
}

TEST(ValueBool, FalseReturnsFalse) {
    EXPECT_EQ(Value{false}.as_bool(), false);
}

// =============================================================================
// Grupo 7: Construtor string
// =============================================================================

TEST(ValueString, IsString) {
    EXPECT_TRUE(Value{std::string("hello")}.is_string());
}

TEST(ValueString, FromCharPtr) {
    Value v{"world"};
    EXPECT_TRUE(v.is_string());
    EXPECT_EQ(v.as_string(), "world");
}

TEST(ValueString, EmptyString) {
    Value v{std::string("")};
    EXPECT_TRUE(v.is_string());
    EXPECT_TRUE(v.as_string().empty());
}

TEST(ValueString, LongString) {
    std::string s(10000, 'A');
    Value v{s};
    EXPECT_EQ(v.as_string(), s);
}

TEST(ValueString, SpecialChars) {
    std::string s = "hello\nworld\t!@#$%^&*()";
    EXPECT_EQ(Value{s}.as_string(), s);
}

TEST(ValueString, UnicodeString) {
    std::string s = "olá mundo 日本語";
    EXPECT_EQ(Value{s}.as_string(), s);
}

// =============================================================================
// Grupo 8: Construtor Array
// =============================================================================

TEST(ValueArray, IsArray) {
    Array arr{Value{1.0}, Value{2.0}};
    EXPECT_TRUE(Value{arr}.is_array());
}

TEST(ValueArray, EmptyArray) {
    Value v{Array{}};
    EXPECT_TRUE(v.is_array());
    EXPECT_TRUE(v.as_array().empty());
}

TEST(ValueArray, SizePreserved) {
    Array arr{Value{1.0}, Value{2.0}, Value{3.0}};
    Value v{arr};
    EXPECT_EQ(v.as_array().size(), 3u);
}

TEST(ValueArray, ElementsAccessible) {
    Array arr{Value{10.0}, Value{20.0}, Value{30.0}};
    Value v{std::move(arr)};
    const auto& a = v.as_array();
    EXPECT_DOUBLE_EQ(a[0].as_number(), 10.0);
    EXPECT_DOUBLE_EQ(a[1].as_number(), 20.0);
    EXPECT_DOUBLE_EQ(a[2].as_number(), 30.0);
}

TEST(ValueArray, MixedTypes) {
    Array arr{Value{1.0}, Value{true}, Value{std::string("hi")}, Value{}};
    Value v{std::move(arr)};
    const auto& a = v.as_array();
    EXPECT_TRUE(a[0].is_number());
    EXPECT_TRUE(a[1].is_bool());
    EXPECT_TRUE(a[2].is_string());
    EXPECT_TRUE(a[3].is_null());
}

TEST(ValueArray, LargeArray) {
    Array arr;
    arr.reserve(100000);
    for (int i = 0; i < 100000; ++i)
        arr.push_back(Value{static_cast<double>(i)});
    Value v{std::move(arr)};
    EXPECT_EQ(v.as_array().size(), 100000u);
}

// =============================================================================
// Grupo 9: Construtor Object
// =============================================================================

TEST(ValueObject, IsObject) {
    Object obj{{"key", Value{1.0}}};
    EXPECT_TRUE(Value{std::move(obj)}.is_object());
}

TEST(ValueObject, EmptyObject) {
    Value v{Object{}};
    EXPECT_TRUE(v.is_object());
    EXPECT_TRUE(v.as_object().empty());
}

TEST(ValueObject, KeysAccessible) {
    Object obj;
    obj["a"] = Value{1.0};
    obj["b"] = Value{std::string("test")};
    Value v{std::move(obj)};
    EXPECT_TRUE(v.as_object().count("a") > 0);
    EXPECT_TRUE(v.as_object().count("b") > 0);
    EXPECT_DOUBLE_EQ(v.as_object().at("a").as_number(), 1.0);
    EXPECT_EQ(v.as_object().at("b").as_string(), "test");
}

TEST(ValueObject, MissingKey) {
    Value v{Object{{"x", Value{1.0}}}};
    EXPECT_EQ(v.as_object().count("missing"), 0u);
}

// =============================================================================
// Grupo 10: Acesso com tipo errado — lança bad_variant_access
// =============================================================================

TEST(ValueWrongType, NumberAsString) {
    Value v{3.14};
    EXPECT_THROW(v.as_string(), std::bad_variant_access);
}

TEST(ValueWrongType, NumberAsBool) {
    Value v{1.0};
    EXPECT_THROW(v.as_bool(), std::bad_variant_access);
}

TEST(ValueWrongType, NumberAsArray) {
    Value v{1.0};
    EXPECT_THROW(v.as_array(), std::bad_variant_access);
}

TEST(ValueWrongType, NumberAsObject) {
    Value v{1.0};
    EXPECT_THROW(v.as_object(), std::bad_variant_access);
}

TEST(ValueWrongType, BoolAsNumber) {
    Value v{true};
    EXPECT_THROW(v.as_number(), std::bad_variant_access);
}

TEST(ValueWrongType, StringAsNumber) {
    Value v{std::string("3.14")};
    EXPECT_THROW(v.as_number(), std::bad_variant_access);
}

TEST(ValueWrongType, ArrayAsObject) {
    Value v{Array{}};
    EXPECT_THROW(v.as_object(), std::bad_variant_access);
}

TEST(ValueWrongType, ObjectAsArray) {
    Value v{Object{}};
    EXPECT_THROW(v.as_array(), std::bad_variant_access);
}

TEST(ValueWrongType, NullAsNumber) {
    Value v;
    EXPECT_THROW(v.as_number(), std::bad_variant_access);
}

// =============================================================================
// Grupo 11: Cópia
// =============================================================================

TEST(ValueCopy, CopyDouble) {
    Value a{42.0};
    Value b = a;
    EXPECT_DOUBLE_EQ(b.as_number(), 42.0);
    // independente
    b = Value{99.0};
    EXPECT_DOUBLE_EQ(a.as_number(), 42.0);
}

TEST(ValueCopy, CopyString) {
    Value a{std::string("original")};
    Value b = a;
    EXPECT_EQ(b.as_string(), "original");
}

TEST(ValueCopy, CopyArray) {
    Array arr{Value{1.0}, Value{2.0}};
    Value a{arr};
    Value b = a;
    EXPECT_EQ(b.as_array().size(), 2u);
}

TEST(ValueCopy, CopyObject) {
    Object obj;
    obj["key"] = Value{7.0};
    Value a{obj};
    Value b = a;
    EXPECT_DOUBLE_EQ(b.as_object().at("key").as_number(), 7.0);
}

// =============================================================================
// Grupo 12: Move
// =============================================================================

TEST(ValueMove, MoveDouble) {
    Value a{3.14};
    Value b = std::move(a);
    EXPECT_DOUBLE_EQ(b.as_number(), 3.14);
}

TEST(ValueMove, MoveString) {
    Value a{std::string("move_me")};
    Value b = std::move(a);
    EXPECT_EQ(b.as_string(), "move_me");
}

TEST(ValueMove, MoveArray) {
    Array arr;
    for (int i = 0; i < 1000; ++i) arr.push_back(Value{static_cast<double>(i)});
    Value a{std::move(arr)};
    Value b = std::move(a);
    EXPECT_EQ(b.as_array().size(), 1000u);
}

// =============================================================================
// Grupo 13: Aninhamento — Array de Objects
// =============================================================================

TEST(ValueNested, ArrayOfObjects) {
    Array arr;
    for (int i = 0; i < 3; ++i) {
        Object obj;
        obj["idx"]   = Value{static_cast<double>(i)};
        obj["label"] = Value{std::string("item_" + std::to_string(i))};
        arr.push_back(Value{std::move(obj)});
    }
    Value v{std::move(arr)};
    ASSERT_TRUE(v.is_array());
    ASSERT_EQ(v.as_array().size(), 3u);
    for (int i = 0; i < 3; ++i) {
        const auto& obj = v.as_array()[i].as_object();
        EXPECT_DOUBLE_EQ(obj.at("idx").as_number(), static_cast<double>(i));
        EXPECT_EQ(obj.at("label").as_string(), "item_" + std::to_string(i));
    }
}

// =============================================================================
// Grupo 14: Aninhamento — Object de Arrays
// =============================================================================

TEST(ValueNested, ObjectOfArrays) {
    Object inputs;
    for (const auto& name : {"speed", "position", "acceleration"}) {
        Array arr{Value{1.0}, Value{2.0}, Value{3.0}};
        inputs[name] = Value{std::move(arr)};
    }
    Value v{std::move(inputs)};
    ASSERT_TRUE(v.is_object());
    EXPECT_EQ(v.as_object().size(), 3u);
    EXPECT_EQ(v.as_object().at("speed").as_array().size(), 3u);
}

// =============================================================================
// Grupo 15: Aninhamento — Object de Objects (deep nesting)
// =============================================================================

TEST(ValueNested, DeepObjectNesting) {
    // Depth 5 nesting
    Object deep;
    Object* cur = &deep;
    for (int i = 0; i < 4; ++i) {
        Object child;
        child["depth"] = Value{static_cast<double>(i + 1)};
        (*cur)["child"] = Value{std::move(child)};
        cur = &(*cur)["child"].as_object();
    }

    Value v{std::move(deep)};
    EXPECT_TRUE(v.is_object());
    EXPECT_TRUE(v.as_object().count("child") > 0);
}

// =============================================================================
// Grupo 16: Aninhamento — Array de Arrays
// =============================================================================

TEST(ValueNested, ArrayOfArrays) {
    Array matrix;
    for (int r = 0; r < 4; ++r) {
        Array row;
        for (int c = 0; c < 4; ++c)
            row.push_back(Value{static_cast<double>(r * 4 + c)});
        matrix.push_back(Value{std::move(row)});
    }
    Value v{std::move(matrix)};
    ASSERT_EQ(v.as_array().size(), 4u);
    ASSERT_EQ(v.as_array()[0].as_array().size(), 4u);
    EXPECT_DOUBLE_EQ(v.as_array()[1].as_array()[2].as_number(), 6.0);
}

// =============================================================================
// Grupo 17: Mutação via as_array() / as_object() não-const
// =============================================================================

TEST(ValueMutation, AppendToArray) {
    Value v{Array{Value{1.0}}};
    v.as_array().push_back(Value{2.0});
    ASSERT_EQ(v.as_array().size(), 2u);
    EXPECT_DOUBLE_EQ(v.as_array()[1].as_number(), 2.0);
}

TEST(ValueMutation, InsertIntoObject) {
    Value v{Object{}};
    v.as_object()["new_key"] = Value{99.0};
    EXPECT_DOUBLE_EQ(v.as_object().at("new_key").as_number(), 99.0);
}

TEST(ValueMutation, OverwriteObjectKey) {
    Value v{Object{{"x", Value{1.0}}}};
    v.as_object()["x"] = Value{2.0};
    EXPECT_DOUBLE_EQ(v.as_object().at("x").as_number(), 2.0);
}

// =============================================================================
// Grupo 18: Object como tipo de input (mapa nome → Value)
// =============================================================================

TEST(ObjectAsInput, EmptyObject) {
    Object o;
    EXPECT_TRUE(o.empty());
}

TEST(ObjectAsInput, SingleTensorInput) {
    Object o;
    Array arr;
    arr.reserve(5);
    for (float f : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f})
        arr.push_back(Value{static_cast<double>(f)});
    o["input"] = Value{std::move(arr)};
    EXPECT_EQ(o.size(), 1u);
    EXPECT_EQ(o.at("input").as_array().size(), 5u);
}

TEST(ObjectAsInput, MultiTensorInput) {
    Object o;
    o["obs"]    = Value{Array{Value{1.0}, Value{2.0}}};
    o["action"] = Value{Array{Value{0.5}}};
    o["mask"]   = Value{Array{Value{true}, Value{false}}};
    EXPECT_EQ(o.size(), 3u);
}

// =============================================================================
// Grupo 19: Valores limites de double
// =============================================================================

TEST(ValueEdge, MaxDouble) {
    double v = std::numeric_limits<double>::max();
    EXPECT_DOUBLE_EQ(Value{v}.as_number(), v);
}

TEST(ValueEdge, LowestDouble) {
    double v = std::numeric_limits<double>::lowest();
    EXPECT_DOUBLE_EQ(Value{v}.as_number(), v);
}

TEST(ValueEdge, DenormalDouble) {
    double v = std::numeric_limits<double>::denorm_min();
    EXPECT_DOUBLE_EQ(Value{v}.as_number(), v);
}

TEST(ValueEdge, NegativeZero) {
    Value v{-0.0};
    EXPECT_TRUE(v.is_number());
    EXPECT_DOUBLE_EQ(v.as_number(), 0.0);
}

// =============================================================================
// Grupo 20: Sequências de Values em Array — verificação de conteúdo
// =============================================================================

TEST(ArrayContent, AllNulls) {
    Array arr{Value{}, Value{}, Value{}};
    for (const auto& v : arr) EXPECT_TRUE(v.is_null());
}

TEST(ArrayContent, AllBools) {
    Array arr{Value{true}, Value{false}, Value{true}};
    EXPECT_TRUE(arr[0].as_bool());
    EXPECT_FALSE(arr[1].as_bool());
    EXPECT_TRUE(arr[2].as_bool());
}

TEST(ArrayContent, Heterogeneous) {
    Array arr{Value{1.0}, Value{false}, Value{std::string("hi")}, Value{}};
    EXPECT_TRUE(arr[0].is_number());
    EXPECT_TRUE(arr[1].is_bool());
    EXPECT_TRUE(arr[2].is_string());
    EXPECT_TRUE(arr[3].is_null());
}
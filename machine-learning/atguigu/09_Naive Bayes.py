from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# ====================
# 训练数据
# ====================
train_texts = [
    "免费 中奖 领取 手机",
    "开会 项目 总结 报告",
    "点击 链接 免费 获取",
    "明天 团队 聚餐 地点"
]
train_labels = ["垃圾邮件", "正常邮件", "垃圾邮件", "正常邮件"]

# ====================
# 1) 向量化：词袋模型（计数）
# token_pattern=None + tokenizer=str.split
# -> 告诉 CountVectorizer：用空格切分（与你的 doc.split() 一致）
# ====================
vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None)

X_train = vectorizer.fit_transform(train_texts)

# ====================
# 2) 朴素贝叶斯：多项式 NB
# alpha=1.0 相当于拉普拉斯平滑（你代码里的 +1）
# ====================
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train, train_labels)

# ====================
# 测试预测
# ====================
test_text = "明天 开会 总结 项目"
X_test = vectorizer.transform([test_text])
prediction = clf.predict(X_test)[0]

print(f"文本: [{test_text}]")
print(f"预测分类: {prediction}")  # 预期输出 '正常邮件'
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from crawl import crawl_amazon_product_reviews  # Hàm crawl định nghĩa riêng
from CleanData import Tokenization, StopWord, StemmingPorter, StemmingSnowball, Lemmatization, PosTagging, PosTaggingChart, Contraction, spellchecker, Ner, NerRender, remove_emoji
from GenData import NLPInsert, NLPSplit, NLPSubstitute, NLPSwap, NLPKeyboard, NLPBackTranslate, NLPReserved, NLPSynonym
from vectorize import OnehotEncoding, BagofWord, BagofN_Gram, TF_IDF, Word2Vec, FastText
from training import train_knn, train_decision_tree, train_svm, train_logistic_regression
from sklearn.metrics import classification_report, confusion_matrix
from training import train_knn, train_decision_tree, train_svm, train_logistic_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import openai
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
client = openai.OpenAI(api_key)
st.set_page_config(page_title="NLP PROCESS", layout="centered")
# ========== TAB GIAO DIỆN ==========
tab1, tab2, tab3 = st.tabs(["🚀 Huấn luyện & Dự đoán", " 🤖chatbot"," Đề xuất phim"])
with tab1:

    st.title("NLP PROCESS")

    # ========== 1. Nhập link crawl Amazon ==========
    link_craw = st.text_input("🔗 Nhập link sản phẩm Amazon")

    if st.button("🐞 Crawl"):
        if link_craw:
            try:
                crawl_amazon_product_reviews(link_craw)
                st.success("✅ Crawl thành công!")
            except Exception as e:
                st.error(f"❌ Lỗi khi crawl: {e}")
        else:
            st.warning("⚠️ Vui lòng nhập link sản phẩm!")

    # ========== 2. Đọc file crawl mặc định ==========
    file_path = "amazon_reviews.csv"
    df_crawl = None

    if os.path.exists(file_path):
        try:
            df_crawl = pd.read_csv(file_path)
            st.success("📄 Đã đọc file crawl amazon_reviews.csv")
            st.dataframe(df_crawl)
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file crawl: {e}")

    # ========== 3. Upload file riêng ==========
    st.markdown("---")
    st.subheader("📁 Hoặc tải lên file CSV của bạn")

    uploaded_file = st.file_uploader("Chọn file CSV", type="csv")
    df_upload = None

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success("✅ Đọc file người dùng thành công!")
            st.dataframe(df_upload)
        except Exception as e:
            st.error(f"❌ Lỗi khi đọc file tải lên: {e}")

    # ========== 4. Chọn cột nhãn và bình luận ==========
    st.markdown("---")
    st.subheader("🧩 Chọn cột Rating và Comment")

    if df_crawl is not None or df_upload is not None:
        with st.form("column_selection"):
            st.markdown("### 🐞 File crawl:")
            col_rating_crawl = st.selectbox("🟡 Cột Rating (crawl)", df_crawl.columns if df_crawl is not None else [], key="crawl_rating")
            col_comment_crawl = st.selectbox("🟣 Cột Comment (crawl)", df_crawl.columns if df_crawl is not None else [], key="crawl_comment")

            st.markdown("### 🧾 File tải lên:")
            col_rating_upload = st.selectbox("🟡 Cột Rating (upload)", df_upload.columns if df_upload is not None else [], key="upload_rating")
            col_comment_upload = st.selectbox("🟣 Cột Comment (upload)", df_upload.columns if df_upload is not None else [], key="upload_comment")

            submit_cols = st.form_submit_button("✅ Gộp dữ liệu")

        # ========== 5. Combine 2 dataframe ==========
        if submit_cols:
            try:
                df_crawl_selected = df_crawl[[col_rating_crawl, col_comment_crawl]].rename(columns={col_rating_crawl: "Rating", col_comment_crawl: "Comment"}) if df_crawl is not None else pd.DataFrame(columns=["Rating", "Comment"])
                df_upload_selected = df_upload[[col_rating_upload, col_comment_upload]].rename(columns={col_rating_upload: "Rating", col_comment_upload: "Comment"}) if df_upload is not None else pd.DataFrame(columns=["Rating", "Comment"])

                df_combined = pd.concat([df_crawl_selected, df_upload_selected], ignore_index=True)
                st.session_state["df_combined"] = df_combined  # ✅ Thêm dòng này

                st.success("✅ Gộp dữ liệu thành công!")
                st.dataframe(df_combined)

                st.download_button(
                    label="📥 Tải file tổng hợp CSV",
                    data=df_combined.to_csv(index=False).encode('utf-8'),
                    file_name="combined_reviews.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"❌ Lỗi khi gộp dữ liệu: {e}")
    # ========== 6. LÀM SẠCH DỮ LIỆU ==========
    st.markdown("---")
    st.subheader("🧹 Làm sạch dữ liệu văn bản")

    # ✅ Kiểm tra và gán lại df_combined từ session_state
    if "df_combined" in st.session_state and not st.session_state["df_combined"].empty:
        df_combined = st.session_state["df_combined"]

        # Chọn số câu để xử lý
        num_sentences = st.slider("🔢 Số câu muốn xử lý", min_value=1, max_value=len(df_combined), value=5)

        # Hiển thị các câu đầu tiên
        st.markdown("### 📝 Các câu sẽ được xử lý")
        st.write(df_combined.head(num_sentences)[["Comment"]])

        # Chọn các phương pháp xử lý
        st.markdown("### ⚙️ Chọn phương pháp làm sạch")
        clean_methods = st.multiselect(
            "Chọn các bước làm sạch bạn muốn áp dụng:",
            ["Contraction", "Spellcheck", "Tokenize", "Remove Stopword", 
            "Stemming (Porter)", "Stemming (Snowball)", "Lemmatization","remove_emoji", 
            "POS Tagging", "NER"],
            default=["Contraction"]
        )

        if st.button("🧼 Làm sạch và thay thế"):
            with st.spinner("⏳ Đang xử lý dữ liệu..."):
                cleaned_comments = []

                for i in range(num_sentences):
                    sentence = df_combined.at[i, "Comment"]

                    # Bước 1: Tokenize nếu được chọn
                    if "Tokenize" in clean_methods:
                        sentence_list = Tokenization(sentence)
                    else:
                        sentence_list = [sentence]

                    # Bước 2: Làm sạch từng câu
                    processed = []
                    for sent in sentence_list:
                        text = sent.text if hasattr(sent, "text") else sent

                        if "Contraction" in clean_methods:
                            text = Contraction(text)
                        if "Spellcheck" in clean_methods:
                            text = spellchecker(text)
                        if "Remove Stopword" in clean_methods:
                            text = StopWord(text)
                        if "Stemming (Porter)" in clean_methods:
                            text = StemmingPorter(text)
                        if "Stemming (Snowball)" in clean_methods:
                            text = StemmingSnowball(text)
                        if "remove_emoji" in clean_methods:
                            text = remove_emoji(text)

                        
                        processed.append(text)

                    # Bước 3: Kết hợp lại
                    result_text = " ".join(processed)

                    # Bước 4: Thêm các bước dạng bảng → chuyển thành chuỗi nếu có
                    if "Lemmatization" in clean_methods:
                        df = Lemmatization(result_text)
                        result_text = " ".join(df["Lemmatization"].astype(str))

                    if "POS Tagging" in clean_methods:
                        df = PosTagging(result_text)
                        result_text = " ".join(df["Từ gốc"].astype(str) + "/" + df["Loại từ"])

                    if "NER" in clean_methods:
                        df = Ner(result_text)
                        result_text = " ".join(df["Thực thể"].astype(str) + "/" + df["Gán nhãn"])

                    cleaned_comments.append(result_text)

                # ✅ Cập nhật lại
                df_combined.loc[:num_sentences-1, "Comment"] = cleaned_comments
            st.success("✅ Đã làm sạch và cập nhật văn bản!")
            st.dataframe(df_combined.head(num_sentences))

    else:
        st.warning("⚠️ Bạn cần gộp dữ liệu trước khi làm sạch văn bản.")

    # ========== 7. SINH DỮ LIỆU VĂN BẢN ==========

    st.markdown("---")
    st.subheader("📈 Sinh dữ liệu văn bản (Data Augmentation)")

    if 'df_combined' in locals() and not df_combined.empty:
        # Chọn số dòng để sinh thêm
        num_generate = st.slider("🔢 Số câu muốn sinh dữ liệu", min_value=1, max_value=len(df_combined), value=3)

        st.markdown("### 📃 Các câu được chọn để sinh thêm")
        st.write(df_combined.head(num_generate)[["Comment"]])

        # Chọn phương pháp sinh dữ liệu
        gen_methods = st.multiselect(
            "🛠 Chọn các phương pháp sinh dữ liệu:",
            ["Insert", "Split", "Substitute", "Swap", "Keyboard Error", "Back Translate", "Reserved Token", "Synonym Replace"],
            default=["Insert", "Substitute"]
        )

        if st.button("📊 Sinh dữ liệu mới"):
            with st.spinner("⏳ Đang sinh dữ liệu..."):
                new_rows = []

                for i in range(num_generate):
                    original = df_combined.at[i, "Comment"]

                    if "Insert" in gen_methods:
                        new_rows.append(NLPInsert(original))
                    if "Split" in gen_methods:
                        new_rows.append(NLPSplit(original))
                    if "Substitute" in gen_methods:
                        new_rows.append(NLPSubstitute(original))
                    if "Swap" in gen_methods:
                        new_rows.append(NLPSwap(original))
                    if "Keyboard Error" in gen_methods:
                        new_rows.append(NLPKeyboard(original))
                    if "Back Translate" in gen_methods:
                        new_rows.append(NLPBackTranslate(original))
                    if "Reserved Token" in gen_methods:
                        new_rows.append(NLPReserved(original))
                    if "Synonym Replace" in gen_methods:
                        new_rows.append(NLPSynonym(original))

                # Gán nhãn cho dữ liệu mới (cùng nhãn như bản gốc)
                rating_col = df_combined.columns[0]  # giả định cột đầu là Rating
                new_data = pd.DataFrame({
                    "Rating": [df_combined.at[i % num_generate, "Rating"] for i in range(len(new_rows))],
                    "Comment": new_rows
                })

                # Thêm vào df_combined
                df_combined = pd.concat([df_combined, new_data], ignore_index=True)
                st.success("✅ Đã sinh và thêm dữ liệu mới vào!")
                st.dataframe(df_combined.tail(len(new_rows)))

                # Cập nhật session_state nếu cần dùng lại
                st.session_state["df_combined"] = df_combined

    # ========== 8. VECTOR HÓA VĂN BẢN ==========
    st.markdown("---")
    st.subheader("📐 Vector hóa văn bản")

    if 'df_combined' in st.session_state and not st.session_state['df_combined'].empty:
        df_combined = st.session_state['df_combined']

        num_vector = st.slider("🔢 Số câu muốn vector hóa", min_value=1, max_value=len(df_combined), value=5, key="vec_slider")
        st.markdown("### 📝 Các câu sẽ được vector hóa")
        st.write(df_combined.head(num_vector)[["Comment"]])

        vec_methods = st.multiselect("🧠 Chọn các phương pháp vector hóa:",
            ["One-hot", "Bag of Words", "Bag of N-Gram", "TF-IDF", "Word2Vec", "FastText"], default=["TF-IDF", "Word2Vec"])

        if st.button("📊 Vector hóa văn bản"):
            with st.spinner("⏳ Đang vector hóa..."):
                result_tables = []
                vec_objects = {}
                text_list = df_combined.head(num_vector)["Comment"].tolist()

                for method in vec_methods:
                    if method == "One-hot":
                        vec, obj = OnehotEncoding(text_list)
                    elif method == "Bag of Words":
                        vec, obj = BagofWord(text_list)
                    elif method == "Bag of N-Gram":
                        vec, obj = BagofN_Gram(text_list)
                    elif method == "TF-IDF":
                        vec, obj = TF_IDF(text_list)
                    elif method == "Word2Vec":
                        vec, obj = Word2Vec(text_list)
                    elif method == "FastText":
                        vec, obj = FastText(text_list)
                    else:
                        continue
                    result_tables.append((method, vec))
                    vec_objects[method] = obj

                combined_vectors = pd.concat([v for _, v in result_tables], axis=1)
                st.dataframe(combined_vectors)

                st.session_state["vector_data"] = combined_vectors
                st.session_state["vector_rows"] = len(combined_vectors)
                st.session_state["vec_methods"] = vec_methods
                st.session_state["vec_objects"] = vec_objects

    # ========== 9. HUẤN LUYỆN MÔ HÌNH ==========
    st.markdown("---")
    st.subheader("🧠 Huấn luyện và so sánh các mô hình phân loại")

    if "vector_data" in st.session_state and "df_combined" in st.session_state:
        df_combined = st.session_state["df_combined"]

        label_column = st.selectbox("🟡 Cột nhãn (label)", df_combined.columns, index=0)
        test_size = st.slider("📊 Tỷ lệ test (%)", min_value=10, max_value=50, value=30) / 100

        if st.button("🚀 Huấn luyện tất cả mô hình"):
            with st.spinner("⏳ Đang huấn luyện..."):
                try:
                    X = st.session_state["vector_data"]
                    vector_rows = st.session_state["vector_rows"]
                    y = df_combined[label_column].iloc[:vector_rows]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    models = {
                        "KNN": train_knn,
                        "Decision Tree": train_decision_tree,
                        "SVM": train_svm,
                        "Logistic Regression": train_logistic_regression
                    }

                    results = []
                    trained_models = {}

                    for name, train_func in models.items():
                        model, acc, y_pred = train_func(X_train, y_train, X_test, y_test)
                        results.append((name, acc))
                        trained_models[name] = model

                        st.markdown(f"### 📊 {name}")
                        st.success(f"✅ Accuracy: {acc:.4f}")

                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                        ax.set_title(f"Ma trận nhầm lẫn - {name}")
                        ax.set_xlabel("Dự đoán")
                        ax.set_ylabel("Thực tế")
                        st.pyplot(fig)

                        st.markdown("📋 Báo cáo chi tiết:")
                        st.text(classification_report(y_test, y_pred))

                    # Accuracy chart
                    st.markdown("## 📈 So sánh Accuracy giữa các mô hình")
                    fig_acc, ax_acc = plt.subplots()
                    ax_acc.bar([n for n, _ in results], [a for _, a in results])
                    ax_acc.set_ylabel("Accuracy")
                    ax_acc.set_ylim(0, 1)
                    st.pyplot(fig_acc)

                    best_model_name, best_acc = max(results, key=lambda x: x[1])
                    best_model = trained_models[best_model_name]
                    st.session_state["best_model"] = best_model
                    st.session_state["best_model_name"] = best_model_name
                    st.success(f"🏆 Mô hình tốt nhất: {best_model_name} (Accuracy = {best_acc:.4f})")
                except Exception as e:
                    st.error(f"❌ Lỗi khi huấn luyện: {e}")

    # ========== 10. DỰ ĐOÁN CÂU MỚI ==========
    st.markdown("---")
    st.subheader("🔮 Dự đoán từ mô hình tốt nhất")

    if "best_model" in st.session_state and "vec_objects" in st.session_state:
        user_input = st.text_input("💬 Nhập một câu đánh giá để dự đoán:", key="predict_input")

        try:
            if user_input:
                text_list = [user_input]
                vec_parts = []
                vec_methods = st.session_state["vec_methods"]
                vec_objects = st.session_state["vec_objects"]

                for method in vec_methods:
                    vectorizer = vec_objects[method]
                    if method in ["TF-IDF", "Bag of Words", "Bag of N-Gram"]:
                        vec = vectorizer.transform(text_list)
                        vec_df = pd.DataFrame(vec.toarray(), columns=vectorizer.get_feature_names_out())
                    elif method in ["Word2Vec", "FastText"]:
                        import spacy
                        nlp = spacy.load("en_core_web_sm")
                        tokens = [token.text for token in nlp(user_input)]
                        vec_np = np.mean([vectorizer.wv[word] for word in tokens if word in vectorizer.wv] or [np.zeros(100)], axis=0)
                        vec_df = pd.DataFrame([vec_np])
                    else:
                        continue
                    vec_parts.append(vec_df)

                input_vector = pd.concat(vec_parts, axis=1)
                input_vector = input_vector.reindex(columns=st.session_state["vector_data"].columns, fill_value=0)

                predicted_label = st.session_state["best_model"].predict(input_vector)[0]
                st.success(f"🧠 Nhãn dự đoán: **{predicted_label}**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
    else:
        st.info("ℹ️ Bạn cần huấn luyện mô hình trước để sử dụng chức năng dự đoán.")




# 👉 Tab giao diện chatbot
with tab2:
    st.title("DUONGPT - Trợ lý ảo thông minh")
    # 1. Khởi tạo lịch sử trò chuyện nếu chưa có
    user_input = st.chat_input("Nhập nội dung trò chuyện...")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "Chào bạn tôi là DUONGPT - một trợ lý ảo thông minh."}
        ]

    # 2. Hiển thị tin nhắn trong khung cuộn giới hạn chiều cao
    with st.container():
        chat_box = st.container()
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"] if msg["role"] in ["user", "assistant"] else "assistant"):
                    st.markdown(msg["content"])

    # 3. Ô nhập nằm ở dưới cùng (luôn xuất hiện sau phần chat)
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state.chat_history
        )
        reply = response.choices[0].message.content

        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
with tab3:
    # Load dữ liệu
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    # Gộp để có tên phim
    df = pd.merge(ratings, movies, on="movieId")

    # Pivot: User-Movie Matrix
    user_movie_matrix = df.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    # Chuẩn hóa
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(user_movie_matrix)

    # Tính cosine similarity giữa phim
    similarity = cosine_similarity(matrix_scaled.T)  # transpose để so phim
    similarity_df = pd.DataFrame(similarity, index=user_movie_matrix.columns, columns=user_movie_matrix.columns)

    # Giao diện Streamlit
    st.title("🎬 Movie Recommender")

    selected_movies = st.multiselect("Chọn phim bạn đã xem:", user_movie_matrix.columns)

    user_ratings = {}
    for movie in selected_movies:
        rating = st.slider(f"Đánh giá cho phim '{movie}'", 1.0, 5.0, 3.0)
        user_ratings[movie] = rating

    if st.button("Đề xuất phim"):
        scores = pd.Series(dtype="float64")
        for movie, rating in user_ratings.items():
            sim_scores = similarity_df[movie] * (rating - 2.5)  # Điều chỉnh trung bình
            scores = scores.add(sim_scores, fill_value=0)

        recommended = scores.sort_values(ascending=False)
        recommended = recommended.drop(labels=selected_movies)
        st.subheader("🎯 Các phim đề xuất:")
        top_movies = recommended.head(10).index.tolist()
        recommendation_info = movies[movies['title'].isin(top_movies)][['title', 'genres']]

        # Gộp theo đúng thứ tự đề xuất
        recommendation_info = recommendation_info.set_index('title').loc[top_movies].reset_index()

        for idx, row in recommendation_info.iterrows():
            st.write(f"🎬 **{row['title']}** — _{row['genres']}_")

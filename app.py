import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 로지스틱 회귀 전에 값 범위를 맞추는 도구
from sklearn.linear_model import LogisticRegression  # 탭3에서 쓸 모델
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)  # 탭4에서 쓸 평가 지표들

st.set_page_config(page_title="반도체 센서 데이터로 불량 여부 예측하기")

st.title("반도체 센서 데이터로 불량 여부 예측하기")
st.caption("SECOM 센서 측정값으로 양품/불량을 판별합니다")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["데이터 훑기", "전처리", "학습", "결과", "리포트"])

with tab1:
    uploaded_file = st.file_uploader("CSV 파일을 올려주세요", type="csv")

    if uploaded_file is None:
        st.write("파일을 올려주세요")
    else:
        df = pd.read_csv(uploaded_file)

        # 1) 행 수와 열 수
        st.write(f"행 {df.shape[0]}개, 열 {df.shape[1]}개")

        # 2) 앞의 다섯 줄
        st.dataframe(df.head())

        # 3) 빈칸 개수
        na_counts = df.isna().sum()
        total_na = int(na_counts.sum())
        st.write(f"빈칸 {total_na}개")
        if total_na > 0:
            na_cols = na_counts[na_counts > 0]
            na_table = pd.DataFrame({
                "열 이름": na_cols.index,
                "빈칸 개수": na_cols.values,
                "빈칸 비율(%)": (na_cols.values / len(df) * 100).round(2),
            })
            st.dataframe(na_table)
        else:
            st.write("빈칸 없음")

        # 4) 결과 열 선택
        st.write("맞는 열인지 확인하세요")
        columns = list(df.columns)
        selected_col = st.selectbox("결과 열을 고르세요", columns, index=len(columns) - 1)
        value_counts = df[selected_col].value_counts()
        value_ratio = df[selected_col].value_counts(normalize=True) * 100
        result_table = pd.DataFrame({
            "값": value_counts.index,
            "개수": value_counts.values,
            "비율(%)": value_ratio.values.round(2),
        })
        st.dataframe(result_table)

with tab2:
    if uploaded_file is None:
        st.write("먼저 '데이터 훑기' 탭에서 파일을 올려주세요")
    else:
        # 빈칸 개수를 맨 위에 먼저 표시
        na_count_before = int(df.isna().sum().sum())
        st.write(f"빈칸 {na_count_before}개")

        if na_count_before == 0:
            st.write("빈칸이 없습니다. 채울 것이 없어요")
            fill_method = None
        else:
            fill_method = st.selectbox("빈칸을 무엇으로 채울까요?", ["중앙값", "평균", "0"])

        # 글자로 된 열 찾기 (결과 열은 따로 처리하므로 제외)
        # pandas 3.x부터 글자 열이 object가 아닌 str dtype으로 읽히기도 해서
        # dtype 비교 대신 "숫자가 아니면 글자 열"로 판단한다 (날짜/시각 열도 여기서 걸러짐)
        text_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c != selected_col
        ]
        if text_cols:
            st.write(f"글자로 된 열: {', '.join(text_cols)}")
            text_action = st.selectbox("글자 열을 어떻게 할까요?", ["학습에서 빼기", "숫자로 바꾸기"])
        else:
            text_action = None

        # 결과 열에서 1로 볼 값 고르기
        pos_value = st.selectbox(
            f"'{selected_col}' 열에서 어떤 값을 1로 볼까요?",
            df[selected_col].dropna().unique().tolist(),
        )

        # 학습용/시험용 나누는 비율
        train_ratio = st.slider("학습용 비율(%)", min_value=50, max_value=90, value=80, step=5)

        if st.button("적용"):
            df_proc = df.copy()

            # 숫자 열의 빈칸 채우기
            numeric_cols = df_proc.select_dtypes(include="number").columns.tolist()
            if fill_method == "중앙값":
                df_proc[numeric_cols] = df_proc[numeric_cols].fillna(df_proc[numeric_cols].median())
            elif fill_method == "평균":
                df_proc[numeric_cols] = df_proc[numeric_cols].fillna(df_proc[numeric_cols].mean())
            elif fill_method == "0":
                df_proc[numeric_cols] = df_proc[numeric_cols].fillna(0)

            # 글자 열 처리
            if text_cols:
                if text_action == "학습에서 빼기":
                    df_proc = df_proc.drop(columns=text_cols)
                    text_summary = f"글자 열 {len(text_cols)}개를 학습에서 뺐습니다"
                else:
                    for c in text_cols:
                        df_proc[c] = df_proc[c].fillna("빈칸").astype("category").cat.codes
                    text_summary = f"글자 열 {len(text_cols)}개를 숫자로 바꿨습니다"
            else:
                text_summary = "글자로 된 열이 없습니다"

            na_count_after = int(df_proc.isna().sum().sum())

            # 결과 열을 0/1로 바꾸고 학습/시험용으로 나누기
            y = (df[selected_col] == pos_value).astype(int)
            X = df_proc.drop(columns=[selected_col])
            test_ratio = (100 - train_ratio) / 100
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_ratio, stratify=y, random_state=42
            )

            st.session_state["prep_result"] = {
                "na_before": na_count_before,
                "na_after": na_count_after,
                "text_summary": text_summary,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "train_pos": int(y_train.sum()),
                "train_pos_ratio": round(float(y_train.mean()) * 100, 2),
                "test_pos": int(y_test.sum()),
                "test_pos_ratio": round(float(y_test.mean()) * 100, 2),
            }

            # 탭3(학습)에서 그대로 쓸 수 있게 실제 학습/시험용 데이터를 저장
            # (전처리를 다시 누르면 이전 학습 결과는 더 이상 맞지 않으므로 함께 지운다)
            st.session_state["split_data"] = {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
            st.session_state.pop("model_result", None)
            st.session_state.pop("eval_result", None)

        if "prep_result" in st.session_state:
            r = st.session_state["prep_result"]
            st.write(f"빈칸 {r['na_before']}개에서 {r['na_after']}개로 줄었습니다")
            st.write(r["text_summary"])
            st.write(f"학습용 행 수 {r['n_train']}개, 시험용 행 수 {r['n_test']}개")
            split_table = pd.DataFrame({
                "구분": ["학습용", "시험용"],
                "1 개수": [r["train_pos"], r["test_pos"]],
                "1 비율(%)": [r["train_pos_ratio"], r["test_pos_ratio"]],
            })
            st.dataframe(split_table)

with tab3:
    if "split_data" not in st.session_state:
        st.write("먼저 '전처리' 탭에서 적용을 눌러주세요")
    else:
        data = st.session_state["split_data"]
        st.write("모델: 로지스틱 회귀 (StandardScaler로 값 범위를 맞춘 뒤 학습)")

        if st.button("학습 시작"):
            # 로지스틱 회귀는 값 범위 차이에 민감하므로 학습용 기준으로 표준화
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(data["X_train"])
            X_test_scaled = scaler.transform(data["X_test"])

            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, data["y_train"])

            st.session_state["model_result"] = {"model": model, "scaler": scaler}
            st.session_state.pop("eval_result", None)  # 새로 학습했으니 이전 평가 결과는 지운다

        if "model_result" in st.session_state:
            st.write("학습이 끝났습니다. '결과' 탭에서 확인하세요")

with tab4:
    if "model_result" not in st.session_state:
        st.write("먼저 '학습' 탭에서 학습을 시작해주세요")
    else:
        data = st.session_state["split_data"]
        mr = st.session_state["model_result"]
        y_test = data["y_test"]

        # 학습 때와 같은 스케일러로 시험용 데이터를 변환한 뒤 예측
        X_test_scaled = mr["scaler"].transform(data["X_test"])
        y_pred = mr["model"].predict(X_test_scaled)

        # 기준 모델: 시험용에서 더 많은 쪽 값으로 항상 답한다고 가정
        majority = int(y_test.mode()[0])
        y_baseline = pd.Series(majority, index=y_test.index)

        compare_table = pd.DataFrame({
            "구분": ["기준 모델(항상 다수값)", "로지스틱 회귀"],
            "정확도": [
                round(accuracy_score(y_test, y_baseline), 4),
                round(accuracy_score(y_test, y_pred), 4),
            ],
            "정밀도": [
                round(precision_score(y_test, y_baseline, zero_division=0), 4),
                round(precision_score(y_test, y_pred, zero_division=0), 4),
            ],
            "재현율": [
                round(recall_score(y_test, y_baseline, zero_division=0), 4),
                round(recall_score(y_test, y_pred, zero_division=0), 4),
            ],
            "F1": [
                round(f1_score(y_test, y_baseline, zero_division=0), 4),
                round(f1_score(y_test, y_pred, zero_division=0), 4),
            ],
        })
        st.write("기준 모델과 나란히 비교")
        st.dataframe(compare_table)

        cm = confusion_matrix(y_test, y_pred)
        cm_table = pd.DataFrame(cm, index=["실제 0", "실제 1"], columns=["예측 0", "예측 1"])
        st.write("혼동행렬")
        st.dataframe(cm_table)

        # 탭5(리포트)에서 다시 계산하지 않도록 결과를 저장
        st.session_state["eval_result"] = {
            "acc": float(compare_table["정확도"][1]),
            "prec": float(compare_table["정밀도"][1]),
            "rec": float(compare_table["재현율"][1]),
            "f1": float(compare_table["F1"][1]),
            "acc_base": float(compare_table["정확도"][0]),
            "majority": majority,
            "n_test": len(y_test),
        }

with tab5:
    if "eval_result" not in st.session_state:
        st.write("먼저 '결과' 탭까지 끝내주세요")
    else:
        r = st.session_state["prep_result"]
        e = st.session_state["eval_result"]

        st.write("### 리포트")
        st.write(f"- 빈칸 {r['na_before']}개 -> {r['na_after']}개로 처리")
        st.write(f"- 학습용 {r['n_train']}개, 시험용 {e['n_test']}개로 분리")
        st.write(
            f"- 기준 모델(항상 {e['majority']}로 답함) 정확도 {e['acc_base']*100:.2f}% "
            f"-> 로지스틱 회귀 정확도 {e['acc']*100:.2f}%"
        )
        st.write(
            f"- 로지스틱 회귀 정밀도 {e['prec']*100:.2f}%, "
            f"재현율 {e['rec']*100:.2f}%, F1 {e['f1']*100:.2f}%"
        )
        st.write(f"- 리포트 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

st.write(f"지금 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

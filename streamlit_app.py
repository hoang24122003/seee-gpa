import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Dự đoán kết quả học tập", layout="centered")
st.markdown(
    "<h3 style='color: blue;'>DỰ ĐOÁN KẾT QUẢ HỌC TẬP SINH VIÊN</h3>",
    unsafe_allow_html=True
)

# =============================
# Load models trước để xác định các kỳ khả dụng
# =============================
# Chọn loại sinh viên và xác định prefix để load đúng file
student_type = st.sidebar.selectbox("Loại sinh viên:", ("8 kỳ", "10 kỳ"))
prefix = '8' if student_type == '8 kỳ' else '10'

# Load model Final CPA dict
path_cpa = f"models_streamlit/final_cpa_{prefix}_ki.joblib"
try:
    cpa_dict = joblib.load(path_cpa)
except Exception as e:
    st.error(f"❌ Không thể load model CPA: {e}")
    st.stop()

# Xác định số kỳ khả dụng từ key của cpa_dict (ví dụ 'GPA_TC_1_2' -> 2 kỳ)
term_counts = {k: len(k.split('_'))-2 for k in cpa_dict.keys()}
available_terms = sorted(set(term_counts.values()))
# =============================
# Sidebar: Chọn kỳ hiện tại dựa vào model khả dụng
# =============================
st.sidebar.subheader("Cài đặt đầu vào")
current_semester = st.sidebar.selectbox("Kỳ hiện tại:", available_terms)

# Sinh key đúng theo lựa chọn
# Lấy key tương ứng với current_semester
group_key = next(k for k,v in term_counts.items() if v == current_semester)

# Nhập GPA và Tín chỉ cho mỗi kỳ
gpa_inputs = []
credit_inputs = []
for i in range(1, current_semester + 1):
    gpa = st.sidebar.number_input(
        f"GPA kỳ {i}", min_value=0.0, max_value=4.0, step=0.01, format="%.2f"
    )
    credit = st.sidebar.number_input(
        f"Tín chỉ kỳ {i}", min_value=1, max_value=30, step=1
    )
    gpa_inputs.append(gpa)
    credit_inputs.append(credit)

# Kiểm tra nhập đủ
if any(val is None for val in gpa_inputs+credit_inputs):
    st.warning("⚠️ Vui lòng nhập đủ GPA và tín chỉ.")
else:
    try:
        # Chuẩn bị dữ liệu
        X = np.array(gpa_inputs + credit_inputs).reshape(1, -1)

        # Dự đoán Final CPA
        scaler_cpa = cpa_dict[group_key]['scaler']
        model_cpa  = cpa_dict[group_key]['model']
        Xc = scaler_cpa.transform(X)
        cpa_pred = model_cpa.predict(Xc)[0]
        st.subheader("🎓 Dự đoán CPA tốt nghiệp")
        st.success(f"Final CPA: {cpa_pred:.2f}")

        # Dự đoán GPA kỳ tiếp theo nếu có model
        # Load next_gpa dict
        path_gpa = f"models_streamlit/next_gpa_{prefix}_ki.joblib"
        gpa_dict = joblib.load(path_gpa)
        if group_key in gpa_dict:
            scaler_gpa = gpa_dict[group_key]['scaler']
            model_gpa  = gpa_dict[group_key]['model']
            Xg = scaler_gpa.transform(X)
            gpa_pred = model_gpa.predict(Xg)[0]
            st.subheader(f"📘 Dự đoán GPA kỳ {current_semester+1}")
            st.info(f"GPA dự đoán: {gpa_pred:.2f}")
    except KeyError as ke:
        available = ', '.join(cpa_dict.keys())
        st.error(f"❌ Key '{ke.args[0]}' không tồn tại. Các key khả dụng: {available}")
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")

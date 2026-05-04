import os
import tempfile
import traceback
import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

# LangChain 및 파서 모듈 Import
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Ollama 로컬 LLM 및 체인 Import
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# (중요) Windows 환경 Tesseract 설치 경로 지정 (설치 환경에 맞게 수정하세요)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 웹 서비스 UI 구성
st.title("LangChain 기반 생성형 AI RAG 시스템 (Ollama 다중 파일 버전)")
st.write("---")

# 수정 1: 다중 파일 업로드 허용 (accept_multiple_files=True)
uploaded_files = st.file_uploader(
    "여러 개의 문서를 한 번에 선택하세요 (PDF, 엑셀, 워드, 이미지 등)",
    type=["pdf", "csv", "xlsx", "xls", "txt", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)
st.write("---")


def process_uploaded_files(files):
    if not files:
        return None

    all_extracted_data = []  # 모든 파일의 텍스트 데이터를 모을 리스트

    # 수정 2: 업로드된 모든 파일을 순회하며 텍스트 추출
    for uploaded_file in files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        # 각 파일별 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            data = []
            # 파일 포맷별 텍스트 추출 라우팅
            if file_extension == ".pdf":
                reader = PdfReader(temp_file_path)
                text_test = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

                if text_test.strip():
                    data = [Document(page_content=page.extract_text(), metadata={"source": uploaded_file.name}) for page
                            in reader.pages if page.extract_text()]
                else:
                    st.info(f"[{uploaded_file.name}] 스캔본 PDF 감지. OCR을 수행합니다...")
                    images = convert_from_path(temp_file_path)
                    ocr_text = ""
                    for img in images:
                        ocr_text += pytesseract.image_to_string(img, lang='kor+eng') + "\n"
                    data = [Document(page_content=ocr_text, metadata={"source": uploaded_file.name})]

            elif file_extension in [".png", ".jpg", ".jpeg"]:
                st.info(f"[{uploaded_file.name}] 이미지 OCR 추출 중...")
                img = Image.open(temp_file_path)
                text = pytesseract.image_to_string(img, lang='kor+eng')
                data = [Document(page_content=text, metadata={"source": uploaded_file.name})]

            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path, encoding='utf-8')
                data = loader.load()
                for doc in data: doc.metadata["source"] = uploaded_file.name

            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
                data = loader.load()
                for doc in data: doc.metadata["source"] = uploaded_file.name

            elif file_extension == ".csv":
                loader = CSVLoader(temp_file_path, encoding='utf-8')
                data = loader.load()
                for doc in data: doc.metadata["source"] = uploaded_file.name

            # 기존: elif file_extension == ".xlsx":
            elif file_extension in [".xlsx", ".xls"]:  # .xls 확장자도 함께 처리하도록 묶음
                # Pandas는 내부적으로 .xlsx면 openpyxl을, .xls면 xlrd를 알아서 호출합니다.
                # 기존 코드
                # df = pd.read_excel(temp_file_path)

                # 개선된 코드 (병합된 셀 문제 완화)
                df = pd.read_excel(temp_file_path)
                # 1. 위아래로 병합된 셀(빈칸)을 이전 값으로 채움
                df = df.ffill(axis=0)
                # 2. 좌우로 병합된 셀(빈칸)을 이전 값으로 채움
                df = df.ffill(axis=1)

                text = "\n".join(
                    df.apply(lambda row: " | ".join([f"{col}: {val}" for col, val in row.items()]), axis=1))
                data = [Document(page_content=text, metadata={"source": uploaded_file.name})]

            # 추출 성공한 데이터를 종합 리스트에 추가
            if data and any(doc.page_content.strip() for doc in data):
                all_extracted_data.extend(data)
            else:
                st.warning(f"[{uploaded_file.name}] 에서 텍스트를 추출하지 못했습니다.")

        except Exception as e:
            st.error(f"[{uploaded_file.name}] 처리 중 오류 발생: {str(e)}")
        finally:
            os.unlink(temp_file_path)  # 임시 파일 삭제

    # 전체 파일 추출 완료 후 청크 분할 및 임베딩 진행
    try:
        if not all_extracted_data:
            st.error("분석할 수 있는 텍스트가 없습니다.")
            return None

        st.info("모든 문서의 텍스트 추출 완료. 벡터 데이터베이스를 구축합니다...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(all_extracted_data)

        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        persist_directory = "./chroma_db"
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings,
                                            persist_directory=persist_directory)

        st.success(f"총 {len(files)}개의 문서가 {len(all_splits)}개의 청크로 병합되어 저장되었습니다.")
        return vectorstore
    except Exception as e:
        st.error(f"벡터 변환 중 오류 발생: {traceback.format_exc()}")
        return None


def load_existing_vectorstore():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
        persist_directory = "./chroma_db"
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            st.info(f"기존 벡터스토어를 불러왔습니다. (현재 총 청크 수: {vectorstore._collection.count()})")
            return vectorstore
        else:
            return None
    except Exception as e:
        st.error(f"기존 벡터스토어 로드 중 오류 발생: {traceback.format_exc()}")
        return None


# 앱 시작 시 기존 벡터스토어 로드
db = load_existing_vectorstore()

# 파일 업로드 시 처리 (업로드 버튼 및 실행 로직)
if uploaded_files:
    if st.button("업로드한 문서들 DB에 저장하기"):
        with st.spinner("문서들을 분석하고 있습니다. 파일이 많을수록 시간이 소요됩니다..."):
            db = process_uploaded_files(uploaded_files)

if db is not None:
    question = st.text_input('문서에 대해 질문을 입력하세요:')

    if st.button('질문하기'):
        with st.spinner('로컬 Ollama AI가 답변을 생성하는 중입니다...'):
            try:
                llm = OllamaLLM(
                    model="gpt-oss:20b",
                    temperature=0.1
                )

                prompt = PromptTemplate.from_template(
                    """
                    주어진 문맥을 바탕으로 질문에 답하세요.
                    문맥: {context}
                    질문: {question}
                    답변:
                    """
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=db.as_retriever(search_kwargs={"k": 10}),
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )

                result = qa_chain.invoke({"query": question})

                # 출처(metadata)를 포함하여 디버깅용 문서 표시
                if "source_documents" in result:
                    with st.expander("검색된 참고 문서 확인하기"):
                        for i, doc in enumerate(result["source_documents"]):
                            source = doc.metadata.get("source", "알 수 없음")
                            st.markdown(f"**[{i + 1}] 출처: {source}**")
                            st.write(f"{doc.page_content[:200]}...")
                            st.write("---")

                st.info(result.get("answer") or result.get("result") or str(result))

            except Exception as e:
                st.error(f"답변 생성 중 오류 발생: {traceback.format_exc()}")
else:
    st.info("문서를 업로드하거나 기존 벡터스토어가 있어야 질문을 할 수 있습니다.")
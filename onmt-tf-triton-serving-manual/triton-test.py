import tritonclient.http as httpclient
import numpy as np
import sentencepiece as spm
import os


class SentencePieceTranslator:
    def __init__(self, triton_url="localhost:8000", model_name=None, sp_model_path=None):
        self.triton_client = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name

        # SentencePiece 모델 로드
        self.sp = None
        if sp_model_path and os.path.exists(sp_model_path):
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(sp_model_path)
            print(f"✅ SentencePiece 모델 로드됨: {sp_model_path}")
        else:
            print("⚠️  SentencePiece 모델을 찾을 수 없습니다. 수동 토큰화 사용.")

    def encode_with_sentencepiece(self, text):
        """텍스트를 토큰으로 인코딩"""
        if self.sp:
            token_pieces = self.sp.encode(text, out_type=str)
            print(f"토큰: {token_pieces}")
            return token_pieces
        else:
            # 수동 토큰화
            tokens = ['▁' + word for word in text.split()]
            return tokens

    def decode_with_sentencepiece(self, token_pieces):
        """토큰을 텍스트로 디코딩"""
        if self.sp and token_pieces:
            try:
                return self.sp.decode(token_pieces)
            except:
                return ' '.join(token_pieces).replace('▁', ' ').strip()
        else:
            return ' '.join(token_pieces).replace('▁', ' ').strip()

    def translate_korean_to_english(self, korean_text):
        """한국어를 영어로 번역"""
        try:
            print(f"🇰🇷 입력: {korean_text}")

            # 토큰화
            token_pieces = self.encode_with_sentencepiece(korean_text)
            tokens_bytes = [piece.encode('utf-8') for piece in token_pieces]
            tokens_array = np.array([tokens_bytes], dtype=object)
            length_array = np.array([[len(token_pieces)]], dtype=np.int32)

            # Triton 입력 준비
            inputs = [
                httpclient.InferInput("tokens", tokens_array.shape, datatype="BYTES"),
                httpclient.InferInput("length", length_array.shape, datatype="INT32")
            ]
            inputs[0].set_data_from_numpy(tokens_array, binary_data=True)
            inputs[1].set_data_from_numpy(length_array, binary_data=True)

            # 출력 설정
            outputs = [
                httpclient.InferRequestedOutput("tokens", binary_data=True),
                httpclient.InferRequestedOutput("length", binary_data=True),
                httpclient.InferRequestedOutput("log_probs", binary_data=True)
            ]

            # 추론 수행
            results = self.triton_client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs
            )

            # 결과 처리
            output_tokens = results.as_numpy("tokens")
            log_probs = results.as_numpy("log_probs")

            # 출력 토큰 추출 및 디코딩
            decoded_tokens = self.extract_output_tokens(output_tokens)
            english_text = self.decode_with_sentencepiece(decoded_tokens)

            print(f"🇺🇸 번역: {english_text}")

            return {
                'korean': korean_text,
                'english': english_text,
                'confidence': float(np.exp(log_probs.mean()))
            }

        except Exception as e:
            print(f"❌ 번역 실패: {e}")
            return None

    def extract_output_tokens(self, output_tokens):
        """출력 배열에서 토큰 문자열 추출"""
        tokens = []
        try:
            # 다차원 배열을 평면화하여 처리
            flat_tokens = output_tokens.flatten()
            for token in flat_tokens:
                if isinstance(token, bytes) and token:
                    try:
                        decoded = token.decode('utf-8')
                        if decoded and decoded not in ['<s>', '</s>', '<pad>', '<unk>', '']:
                            tokens.append(decoded)
                    except:
                        continue
        except Exception as e:
            print(f"토큰 추출 오류: {e}")
        return tokens


def get_user_inputs():
    """사용자로부터 설정값 입력받기"""
    print("=" * 50)
    print("🚀 SentencePiece Translator 설정")
    print("=" * 50)

    # Triton 서버 URL
    triton_url = "61.252.58.171:18000"

    # 모델 이름
    model_name = input("Triton 모델 이름: ").strip()
    if not model_name:
        print("❌ 모델 이름은 필수입니다!")
        return None, None, None

    # SentencePiece 모델 경로
    sp_model_path = input("SentencePiece 모델 파일 경로 (.model): ").strip()
    if not sp_model_path:
        sp_model_path = None

    return triton_url, model_name, sp_model_path


def main():
    """메인 함수"""
    # 사용자 입력 받기
    triton_url, model_name, sp_model_path = get_user_inputs()
    if not model_name:
        return

    # 번역기 초기화
    translator = SentencePieceTranslator(
        #triton_url=triton_url,
        model_name=model_name,
        sp_model_path=sp_model_path
    )

    # 테스트 문장들
    test_sentences = [
        "안녕하세요",
        "오늘 날씨가 좋습니다",
        "저는 한국 사람입니다",
        "감사합니다"
    ]

    print("\n" + "=" * 50)
    print("🚀 한국어 → 영어 번역 테스트")
    print("=" * 50)

    successful = 0
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n📝 테스트 {i}/{len(test_sentences)}: {sentence}")
        print("-" * 30)

        result = translator.translate_korean_to_english(sentence)
        if result:
            successful += 1
            print(f"✅ 성공! 신뢰도: {result['confidence']:.4f}")
        else:
            print("❌ 실패")

    print(f"\n🎯 성공률: {successful}/{len(test_sentences)} ({successful / len(test_sentences) * 100:.1f}%)")


if __name__ == "__main__":
    main()

import torch # 파이토치 패키지 임포트
import torch.nn as nn # 자주 사용하는 torch.nn패키지를 별칭 nn으로 명명
from transformers import BertConfig, BertModel
# 허깅페이스의 트랜스포머 패키지에서 BertConfig, BertModel 클래스 임포트

class CategoryClassifier(nn.Module):
    """상품정보를 받아서 대/중/소/세 카테고리를 예측하는 모델    
    """
    def __init__(self, cfg):
      super(CategoryClassifier, self).__init__()

      # 글로벌 설정 값 사용
      self.cfg = cfg
      # 버트모델의 설정값을 멤버 변수로 저장
      self.bert_cfg = BertConfig( 
          cfg.vocab_size, # 사전 크기
          hidden_size=cfg.hidden_size, # 히든 크기
          num_hidden_layers=cfg.nlayers, # 레이어 층 수
          num_attention_heads=cfg.nheads, # 어텐션 헤드의 수
          intermediate_size=cfg.intermediate_size, # 인터미디어트 크기
          hidden_dropout_prob=cfg.dropout, # 히든 드롭아웃 확률 값
          attention_probs_dropout_prob=cfg.dropout, # 어텐션 드롭아웃 확률 값 
          max_position_embeddings=cfg.seq_len, # 포지션 임베딩의 최대 길이
          type_vocab_size=cfg.type_vocab_size, # 타입 사전 크기
      )
      # 텍스트 인코더로 버트모델 사용
      self.text_encoder = BertModel(self.bert_cfg)

      def get_classifiier(target_size):
        return nn.Sequential(
          nn.Linear(cfg.text_encoder_size, cfg.hidden_size),
                    nn.LayerNorm(cfg.hidden_size),
                    nn.Dropout(cfg.dropout),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_size, target_size),
        )

      # 대 카테고리 분류기
      self.main_cls = get_classifiier(cfg.main_ctg)
      # 중 카테고리 분류기
      self.midd_cls = get_classifiier(cfg.midd_ctg)



    def forward(self, token_ids, token_mask, token_types, label=None):
      """        
      매개변수
      token_ids: 전처리된 상품명을 인덱스로 변환하여 token_ids를 만들었음
      token_mask: 실제 token_ids의 개수만큼은 1, 나머지는 0으로 채움
      token_types: ▁ 문자를 기준으로 서로 다른 타입의 토큰임을 타입 인덱스로 저장
      img_feat: resnet50으로 인코딩된 이미지 피처
      label: 정답 대/중/소/세 카테고리
      """
      # 전처리된 상품명을 하나의 텍스트벡터(text_vec)로 변환
      # 반환 튜플(시퀀스 아웃풋, 풀드(pooled) 아웃풋) 중 시퀀스 아웃풋만 사용
      text_output = self.text_encoder(token_ids, token_mask, token_type_ids=token_types)[0]
      
      # 시퀀스 중 첫 타임스탭의 hidden state만 사용. 
      text_vec = text_output[:, 0]




      # 결합된 벡터로 대카테고리 확률분포 예측
      main_pred = self.main_cls(text_vec)
      # 결합된 벡터로 중카테고리 확률분포 예측
      midd_pred = self.midd_cls(text_vec)


      main_label, midd_label = label.split(1,1)

      loss_fuc = nn.CrossEntropyLoss(ignore_index= -1)

    #   loss_fuc.cuda()

      main_loss = loss_fuc(main_pred, main_label.view(-1))
      # 중카테고리의 예측된 확률분포와 정답확률 분포의 차이를 손실로 반환
      midd_loss = loss_fuc(midd_pred, midd_label.view(-1))

      # loss = (1*main_loss + 1.7*midd_loss) / 2

      # 중뷴류 정확도를 높이기 위하여
      loss = midd_loss

      return loss, [main_pred, midd_pred]

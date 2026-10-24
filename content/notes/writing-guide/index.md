---
title: "이 블로그에 기록하는 방법"
date: 2026-10-24
summary: "사진, 코드, 수식과 그래프를 담는 글쓰기 안내."
tags: ["사용 안내"]
math: true
---
## 사진 넣기

글마다 폴더를 만들고, 글 파일 `index.md` 옆에 사진을 넣습니다.

```markdown
![사진에 대한 설명](photo.jpg)
```

대표 이미지는 같은 폴더에 `feature.jpg`라는 이름으로 넣으면 됩니다.

## 코드 넣기

코드 블록에 언어 이름을 붙이면 색상이 자동으로 적용됩니다.

```python
import matplotlib.pyplot as plt

steps = [1, 2, 3, 4]
loss = [0.9, 0.6, 0.4, 0.3]  # 설명을 위한 예시 데이터
plt.plot(steps, loss)
plt.savefig("loss.png", dpi=180, bbox_inches="tight")
```

생성한 `loss.png`도 사진처럼 본문에 넣을 수 있습니다.

## 수식 넣기

글 설정에 `math: true`를 추가하면 수식을 쓸 수 있습니다.

$$ L = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat y_i)^2 $$

## 인터랙티브 그래프

아래는 **기능 확인을 위한 예시 데이터**입니다.

{{< chart >}}
type: 'line',
data: { labels: ['1', '2', '3', '4'], datasets: [{ label: '예시 loss', data: [0.9, 0.6, 0.4, 0.3], borderColor: '#657653' }] },
options: { responsive: true }
{{< /chart >}}

## 논문 리뷰 순서

연구 질문 → 핵심 아이디어 → 방법 → 실험 결과 → 한계 → 나의 생각.

원문 링크와 그림 출처를 함께 남겨두면 나중에 다시 읽기 편합니다.

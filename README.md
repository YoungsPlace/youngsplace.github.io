# Study Notes · 공부 아카이브

블로그: https://youngsplace.github.io/  

사진과 코드, 논문과 실험을 담는 한국어 블로그입니다. Blowfish 기반이며, 검정·아이보리·골드 색감을 사용합니다.

## GitHub에서 글 쓰기

1. `content/local-ai-models-on-a-mac`(스터디), `content/notes`(공부), `content/papers`(논문), `content/journal`(사진) 중 하나를 엽니다.
2. Add file → Create new file을 누르고 `글이름/index.md`로 만듭니다.
3. 아래 형식으로 작성한 뒤 Commit changes로 저장합니다. 자동 게시에 잠시 시간이 걸립니다.

```markdown
---
title: "오늘 배운 것"
date: 2026-10-24
draft: false
summary: "한 줄 소개"
tags: ["공부"]
math: true
---
## 질문

## 이해한 것

## 다음에 확인할 것
```

사진은 index.md와 같은 폴더에 업로드하고 `![사진 설명](photo.jpg)`로 넣습니다. 대표 사진은 `feature.jpg`로 저장합니다. 공개 전인 글은 `draft: true`로 두세요.

논문 리뷰 작성 틀은 `archetypes/papers.md`에 있습니다. 수식·코드·그래프 예시는 블로그의 ‘이 블로그에 기록하는 방법’ 글을 참고하세요.

## 내 컴퓨터에서 미리보기

Hugo Extended 0.167.0을 사용합니다. 저장소를 내려받은 뒤 테마를 준비하세요.

```sh
mkdir -p themes/blowfish
curl -fL https://github.com/nunocoracao/blowfish/archive/ebe683e9c2668972b71bc011e61e0004ff8e7689.tar.gz | tar -xz --strip-components=1 -C themes/blowfish
hugo server -D
```

새 글: `hugo new content papers/논문이름/index.md`. 빌드: `hugo --minify`.

테마는 공식 MIT 라이선스에 따라 사용합니다. 테마 원본과 라이선스: https://github.com/nunocoracao/blowfish

## Local AI Models on a Mac

스터디 기록: https://youngsplace.github.io/local-ai-models-on-a-mac/

1주차 글은 아직 작성하지 않았습니다. 다음에 이 폴더에 첫 글을 추가하면 됩니다.

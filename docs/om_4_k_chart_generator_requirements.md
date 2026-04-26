# osu!mania 4K 自动练习谱生成器需求文档

## 1. 项目概述

### 1.1 项目名称

osu!mania 4K 自动练习谱生成器

### 1.2 项目定位

本项目是一个面向玩家的娱乐向谱面生成工具。用户提供音频文件和可选背景图，通过配置谱面类型、键型、目标难度、分辨率集合、LN 参数和生成数量，自动生成可导入 osu! 的 `.osz` 文件。

项目目标不是生成达到人工谱师审美标准的高质量发布谱，而是生成“看起来过得去、能玩、可用于练习”的 4K 练习谱。

### 1.3 目标用户

目标用户为需要快速生成练习谱面的 osu!mania / 4K 下落式音游玩家。

典型用户包括：

- 想用任意歌曲生成练习谱的玩家；
- 想练习特定键型，例如 jack、stream、speed、tech 的玩家；
- 想批量生成不同难度练习谱的玩家；
- 不追求人类谱师风格，只要求谱面可玩、难度大致符合预期的用户。

### 1.4 核心目标

系统需要完成以下目标：

1. 支持用户上传音频文件；
2. 支持用户上传可选背景图；
3. 自动分析 BPM、offset、静音段、能量变化和 onset；
4. 支持用户手动修正 BPM 和 offset；
5. 支持 4K osu!mania 谱面生成；
6. 支持 rice、LN、hybrid、vibro 四类谱面类型；
7. 支持 jack、tech、speed、stream 四类键型；
8. 支持 hybrid 模式下配置键型比例；
9. 支持多难度批量生成；
10. 输出 `.zip` 文件，内部包含音频、背景图和一个或多个 `.osu` 谱面文件；
11. 用户可将 `.zip` 后缀改为 `.osz` 后导入 osu!。

---

## 2. 项目范围

### 2.1 当前版本范围

第一版只实现 osu!mania 4K。

虽然系统架构需要预留多 key 扩展能力，但 MVP 不实现 5K、6K、7K 或 10K。

### 2.2 不在第一版范围内的功能

以下功能不纳入 MVP：

- 实时谱面预览；
- 内置音频播放器同步预览；
- 在线生成服务；
- 端到端深度学习模型训练；
- 人类谱师风格模仿；
- 多 BPM / 变速支持；
- SV 特效生成；
- mirror / flip 后处理；
- osu!lazer API 直接导入；
- 谱面质量达到 Ranked / Loved 标准。

---

## 3. 技术路线

### 3.1 总体技术路线

采用纯规则生成路线，开发过程中可使用 AI 辅助编码，但成品项目本身不接入 AI API。

推荐架构：

```text
Streamlit UI
  ↓
音频读取与预处理
  ↓
BPM / offset / onset / energy 分析
  ↓
可选音轨分离增强分析
  ↓
候选 note 时间点生成
  ↓
分辨率 / beat grid snapping
  ↓
谱面类型生成器 rice / LN / hybrid / vibro
  ↓
键型生成器 jack / tech / speed / stream
  ↓
质量约束与合法性修复
  ↓
难度估计与迭代校正
  ↓
.osu 文件生成
  ↓
.zip / .osz 打包输出
```

### 3.2 推荐开发栈

#### 前端 / UI

使用 Streamlit。

理由：

- 纯 Python 开发；
- 适合个人一周 MVP；
- 文件上传、参数表单、进度显示、下载按钮实现成本低；
- 不需要额外前端工程能力。

#### 后端 / 核心逻辑

使用 Python。

推荐依赖：

```text
streamlit        UI
librosa          音频分析、onset、tempo
numpy            数值计算
scipy            信号处理
soundfile        音频读取
pydub            音频格式处理，可选
matplotlib       可选调试图
zipfile          打包输出
```

可选增强依赖：

```text
demucs           音轨分离，可选增强模式
```

### 3.3 音轨分离策略

音轨分离不是判断静音段的唯一手段。静音段可通过整体能量阈值检测完成。

第一版应采用两层策略：

#### 默认模式：不启用音轨分离

使用以下特征：

- RMS energy；
- onset strength；
- spectral flux；
- beat grid；
- local density curve。

该模式速度快、依赖少、适合 MVP。

#### 增强模式：启用 Demucs

启用后可将音频分离为：

- drums；
- bass；
- vocals；
- other。

用途：

- drums stem 用于强节奏点检测；
- bass stem 用于低频持续音和 LN 候选；
- vocal / other stem 用于旋律段辅助；
- 总能量和 stem 能量共同用于静音段判断。

注意：Demucs 运行较慢，第一版应作为可选功能，不应作为默认必需流程。

---

## 4. 输入需求

### 4.1 必选输入

用户必须上传一个音频文件。

支持格式：

```text
.mp3
.wav
.ogg
.flac 可选
```

### 4.2 可选输入

用户可上传背景图。

支持格式：

```text
.jpg
.jpeg
.png
```

如果用户未上传背景图，则 `.osu` 文件中不写入背景，或使用默认背景占位。

### 4.3 元数据输入

用户可填写：

- Title；
- Artist；
- Creator；
- Source；
- Tags；
- Version / Difficulty name；
- Audio filename；
- Background filename。

如果用户不填写，系统自动从文件名生成默认值。

---

## 5. BPM、Offset 与时间系统

### 5.1 BPM 检测

系统需要自动检测 BPM。

要求：

- 自动给出 BPM 初始值；
- UI 中允许用户手动修改 BPM；
- 第一版只支持单 BPM；
- 不支持变速、变拍、多 timing section。

### 5.2 Offset 检测

系统需要自动估计 offset。

要求：

- 自动给出 offset 初始值；
- UI 中允许用户手动修改 offset，单位 ms；
- offset 用于建立 beat grid。

### 5.3 时间表示

系统内部所有 note 时间最终都以毫秒表示。

`.osu` 文件中的 HitObject 时间为：

```text
距离音频开始后的毫秒数
```

### 5.4 分辨率与 Snap

谱面文件本身不强制分辨率，但生成器需要使用分辨率约束以保证谱面可读、可玩和音乐性。

系统需要支持以下分辨率集合：

二分体系：

```text
1/1
1/2
1/4
1/8
1/16
```

三连体系：

```text
1/3
1/6
1/12
```

生成流程：

```text
音频分析得到候选时间点
  ↓
基于 BPM 和 offset 建立 beat grid
  ↓
根据用户允许的分辨率集合生成合法 snap 点
  ↓
将候选时间点吸附到最近的合法 snap 点
  ↓
过滤过近、重复或低置信度点
```

### 5.5 自动切换分辨率

系统应允许自动切换分辨率，而不是将难度和分辨率强绑定。

原则：

- 分辨率由音频局部节奏需要决定；
- 难度由最终密度、键型、chord、jack、LN、MSD 估计等共同决定；
- 用户可限制允许使用的分辨率集合；
- 系统在允许集合内自动选择最贴合 onset 的 snap 点。

---

## 6. 谱面类型需求

系统支持四种谱面类型。

### 6.1 Rice / Note 模式

特点：

- 只生成普通 note；
- 不生成 LN；
- 可选择 jack、tech、speed、stream 键型。

适用：

- 常规练习谱；
- 键型专项练习；
- 高密度 note 练习。

### 6.2 Long Note / 面 模式

特点：

- 主要生成 LN；
- 可包含少量普通 note，但不能与 LN 冲突；
- LN 长度由音频持续能量、bass / melody 持续段和用户参数共同决定。

需要配置：

- LN 比例；
- 最小 LN 时长；
- 最大 LN 时长；
- LN 密度；
- 是否允许 LN 尾接 note；
- 是否允许多轨同时 LN。

### 6.3 Hybrid 模式

特点：

- rice 与 LN 混合；
- 支持 jack、tech、speed、stream 键型比例配置；
- 支持普通 note 与 LN 的混合段落。

用户需要配置：

```text
jack 比例
tech 比例
speed 比例
stream 比例
LN 比例
```

系统需要对比例做归一化处理。

### 6.4 Vibro 模式

特点：

- 极高频 jack / trill 型练习；
- 不包含 LN；
- 主要用于 vibro 专项练习。

需要配置：

- vibro 轨道数量；
- vibro 主要列；
- 是否允许双列 vibro；
- 最大连续 vibro 长度；
- vibro 段落密度；
- vibro 段落之间的休息间隔。

---

## 7. 键型需求

键型描述 note 的形状和轨道分配方式，不描述 note 类型。

note 类型由 rice / LN / hybrid / vibro 决定。

### 7.1 Jack / 叠

定义：同一轨道连续出现 note。

示例：

```text
1 1 1 1
```

需求：

- 支持最大 jack 长度限制；
- 支持 jack 段间休息；
- 支持 chordjack；
- 不允许 LN 内同列再出现普通 note。

### 7.2 Tech / 技

定义：包含较多变化、跳列、楼梯、trill、anchor 等技巧型 pattern。

可能 pattern：

```text
1 2 3 4
1 3 2 4
1 2 1 3
2 4 2 3
```

需求：

- 保证局部 pattern 可读；
- 避免完全随机；
- 避免过长单一重复；
- 可混合 stair、trill、anchor、jump。

### 7.3 Speed / 乱

定义：偏随机、高速、分布较散的 note pattern。

需求：

- 使用受约束随机，不允许完全无约束随机；
- 可设置左右手平衡；
- 默认左右手平衡为 0%，表示不强制平衡；
- 仍需遵守最大 jack、最大纵连、LN 冲突等硬约束。

### 7.4 Stream / 切

定义：强调连续切换、左右手交替、连续流式 pattern。

示例：

```text
1 2 3 4 1 2 3 4
1 3 2 4 1 3 2 4
```

需求：

- 保持连续性；
- 减少同列重复；
- 支持左右手交替；
- 支持 chord stream。

---

## 8. Chord 与纵连需求

### 8.1 Chord 定义

同一时间点多个轨道同时出现 note，即为 chord。

示例：

```text
[1+3]
[2+4]
[1+2+4]
```

### 8.2 Chord 支持

系统必须允许 chord，否则 jack / chordjack / stream 的很多形式无法生成。

### 8.3 Chord 限制参数

用户可配置：

- 是否允许 chord；
- 最大 chord size；
- chord 出现概率；
- 是否允许 3 键 chord；
- 是否允许 4 键 chord；
- 最大连续 chord 数量；
- chord 与 LN 的冲突策略。

### 8.4 默认建议

MVP 默认：

```text
允许 chord：是
最大 chord size：2
允许 3 键 chord：否
允许 4 键 chord：否
```

---

## 9. LN 合法性需求

### 9.1 LN 与普通 note 冲突

任何情况下都禁止同一轨道在 LN 按住期间出现普通 note。

非法示例：

```text
轨道 1：LN 从 1000ms 到 3000ms
轨道 1：普通 note 在 1800ms
```

该情况必须被检测并修复。

### 9.2 LN 与 LN 冲突

同一轨道不允许重叠 LN。

非法示例：

```text
轨道 2：LN 1000ms-3000ms
轨道 2：LN 2000ms-4000ms
```

### 9.3 LN 修复策略

可选修复方式：

1. 删除冲突普通 note；
2. 缩短 LN；
3. 移动普通 note 到其他轨道；
4. 移动 LN 到其他轨道；
5. 优先保留高置信度对象。

MVP 建议：

```text
优先保留 LN，删除同列冲突普通 note。
```

---

## 10. 难度需求

### 10.1 难度输入

用户只输入目标 SR（Star Rating，星数）。

星数范围不设固定上限，但 UI 应给出常用提示范围，例如：

```text
1.0 - 10.0+
```

### 10.2 SR 与 NPS/KPS 的关系

SR 不应简单等同于 NPS / KPS。

NPS / KPS 只作为内部辅助指标，而不是最终难度定义。

系统应理解：

```text
SR ≈ 密度 + pattern + chord + jack + LN + 手型 + 分辨率 + 局部爆发 + 稳定性
```

### 10.3 难度校正策略

由于 MVP 不实现完整 osu!mania 官方 SR 算法，系统采用启发式 SR 近似控制。

#### 第一阶段：启发式 SR 估计

根据以下特征估计谱面难度：

- notes per second；
- key presses per second；
- chord density；
- max chord size；
- jack density；
- longest jack；
- stream length；
- LN ratio；
- LN overlap pressure；
- hand imbalance；
- peak density；
- average density；
- note spacing variance；
- snap subdivision complexity。

#### 第二阶段：迭代调整

生成后计算估计 SR，如果偏离目标，则调整：

- note 数量；
- chord 概率；
- jack 强度；
- LN 比例；
- 局部爆发密度；
- pattern 复杂度；
- 可用 snap 点密度。

循环次数建议：

```text
最多 3-5 次
```

### 10.4 后续扩展

后续可参考 osu! 开源仓库中的 osu!mania difficulty / performance 相关实现，逐步替换启发式估计器。

MVP 中不要求完全复刻官方 SR，只要求生成结果相对接近用户输入目标，并且不同 SR 配置之间有可感知的难度差异。

---

## 11. 质量控制规则

### 11.1 必须实现规则

MVP 必须实现：

1. 同一轨道同一时间不能有多个对象；
2. LN 内不能出现同轨普通 note；
3. 同轨 LN 不能重叠；
4. 最大 jack 长度限制；
5. 最大纵连长度限制；
6. 最小 note 间隔限制；
7. 静音段禁止生成普通 note；
8. vibro 模式禁止 LN；
9. rice 模式禁止 LN；
10. 生成对象必须 snap 到允许的分辨率集合；
11. `.osu` 文件格式必须合法。

### 11.2 推荐实现规则

建议实现：

1. 受约束随机列分配；
2. 左右手平衡控制；
3. chord size 限制；
4. 最大连续 chord 限制；
5. 局部密度上限；
6. 局部休息段插入；
7. 低能量段降密度；
8. 高能量段提升密度；
9. pattern 多样性约束；
10. 避免整首歌单一 pattern。

### 11.3 可选规则

可后续实现：

1. hand strain 估计；
2. finger control 估计；
3. anchor 过载检测；
4. chordjack 爆发检测；
5. LN 尾判定优化；
6. 近似 MSD 计算；
7. 近似 osu!mania star rating 计算。

---

## 12. 生成流程设计

### 12.1 单个难度生成流程

```text
读取用户配置
  ↓
读取音频
  ↓
分析 BPM / offset / energy / onset
  ↓
生成 beat grid
  ↓
生成可用 snap 点
  ↓
过滤静音段
  ↓
根据谱面类型选择生成器
  ↓
根据键型生成轨道 pattern
  ↓
生成普通 note / LN / chord
  ↓
合法性检查
  ↓
难度估计
  ↓
若偏离目标则调整参数重新生成
  ↓
导出 .osu
```

### 12.2 多难度生成流程

用户可选择生成多个难度。

每个难度单独配置：

- Version 名称；
- 目标星数；
- 目标 MSD；
- 谱面类型；
- 键型；
- hybrid 权重；
- LN 参数；
- chord 参数；
- 分辨率集合；
- 质量约束。

系统逐个生成 `.osu` 文件，并统一打包。

---

## 13. UI 需求

### 13.1 页面结构

Streamlit UI 分为以下区域：

1. 文件上传区；
2. 元数据填写区；
3. BPM / offset 区；
4. 难度配置区；
5. 谱面类型配置区；
6. 键型配置区；
7. 质量约束配置区；
8. 生成按钮；
9. 日志 / 进度区；
10. 下载区。

### 13.2 文件上传区

包含：

- 音频上传；
- 背景图上传；
- 音频文件名显示；
- 背景图文件名显示。

### 13.3 BPM / Offset 区

包含：

- 自动检测按钮；
- BPM 输入框；
- offset 输入框；
- 是否启用音轨分离选项。

### 13.4 多难度配置

支持用户添加多个难度配置。

每个难度包含：

- Version 名称；
- 目标星数；
- 目标 MSD；
- 谱面类型；
- 键型；
- 分辨率集合；
- chord 设置；
- LN 设置；
- 特殊模式设置。

### 13.5 谱面类型 UI 逻辑

#### Rice

显示：

- 键型选择；
- chord 参数；
- jack / stream / speed / tech 参数；
- 分辨率集合。

隐藏：

- LN 参数。

#### LN

显示：

- LN 参数；
- chord 参数；
- 分辨率集合。

#### Hybrid

显示：

- jack / tech / speed / stream 权重；
- LN 比例；
- chord 参数；
- 分辨率集合。

#### Vibro

显示：

- vibro 专属参数；
- 分辨率集合；
- 最大连续 vibro 长度。

隐藏：

- LN 参数。

---

## 14. 输出与 `.osu` 文件格式需求

### 14.1 输出格式

系统输出一个 `.zip` 文件。

文件结构：

```text
output.zip
  ├── audio.mp3
  ├── background.jpg
  ├── Artist - Title (Creator) [Difficulty 1].osu
  ├── Artist - Title (Creator) [Difficulty 2].osu
  └── Artist - Title (Creator) [Difficulty 3].osu
```

用户可将 `.zip` 后缀改为 `.osz`。

### 14.2 `.osu` 文件版本

MVP 固定输出：

```text
osu file format v14
```

理由：

- `v14` 是 osu! stable 中常见且兼容性较好的格式；
- 上传样例之一使用 `v14`，另一个使用 `v128`；
- `v128` 更偏 lazer 生态，第一版不强依赖；
- 固定输出 `v14` 可降低兼容性和字段差异风险。

### 14.3 `.osu` 文件章节顺序

生成文件必须包含以下章节，并建议按此顺序输出：

```text
osu file format v14

[General]
[Editor]
[Metadata]
[Difficulty]
[Events]
[TimingPoints]
[Colours]      可选
[HitObjects]
```

根据 osu! wiki，`.osu` 文件由多个方括号章节组成，包括 `[General]`、`[Editor]`、`[Metadata]`、`[Difficulty]`、`[Events]`、`[TimingPoints]`、`[Colours]`、`[HitObjects]` 等。文件第一行声明格式版本。上传样例也遵循这一结构。

### 14.4 `[General]` 要求

MVP 必须写入：

```text
[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: -1
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
SpecialStyle: 0
WidescreenStoryboard: 0
```

字段要求：

- `AudioFilename` 必须与 zip 内音频文件名完全一致；
- `Mode` 必须为 `3`，表示 osu!mania；
- `Countdown` 建议为 `0`；
- `SpecialStyle` 第一版固定为 `0`；
- `WidescreenStoryboard` 可根据是否有背景图设置，默认 `0`。

### 14.5 `[Editor]` 要求

MVP 可写入：

```text
[Editor]
DistanceSpacing: 1
BeatDivisor: 4
GridSize: 4
TimelineZoom: 1
```

说明：

- `BeatDivisor` 仅影响编辑器显示，不决定实际 note 时间；
- 实际 note 时间仍由毫秒时间戳决定；
- 用户选择的分辨率集合用于生成和 snap，而不是写入单一 `BeatDivisor`。

### 14.6 `[Metadata]` 要求

MVP 必须写入：

```text
[Metadata]
Title:<title>
TitleUnicode:<title_unicode>
Artist:<artist>
ArtistUnicode:<artist_unicode>
Creator:<creator>
Version:<difficulty_name>
Source:<source>
Tags:<tags>
BeatmapID:0
BeatmapSetID:-1
```

要求：

- 每个难度文件必须拥有不同的 `Version`；
- 自动生成谱面默认 `BeatmapID:0`；
- 自动生成谱面默认 `BeatmapSetID:-1`；
- 如果用户未填写 Unicode 字段，则与普通字段相同。

### 14.7 `[Difficulty]` 要求

MVP 固定输出 4K mania 难度设置：

```text
[Difficulty]
HPDrainRate:<hp>
CircleSize:4
OverallDifficulty:<od>
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1
```

要求：

- `CircleSize` 必须为 `4`，表示 4K；
- `HPDrainRate` 可由用户设置，默认 `8`；
- `OverallDifficulty` 可由用户设置或根据 SR 估计，默认 `8`；
- `ApproachRate` 对 mania 实际意义较弱，默认 `5`；
- `SliderMultiplier` 和 `SliderTickRate` 对 mania note 生成影响较小，但应保持合法默认值。

### 14.8 `[Events]` 要求

如果用户上传背景图，写入：

```text
[Events]
//Background and Video events
0,0,"background.jpg",0,0
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Layer 4 (Overlay)
//Storyboard Sound Samples
```

如果用户未上传背景图，可写入空事件章节：

```text
[Events]
//Background and Video events
//Break Periods
//Storyboard Layer 0 (Background)
//Storyboard Layer 1 (Fail)
//Storyboard Layer 2 (Pass)
//Storyboard Layer 3 (Foreground)
//Storyboard Layer 4 (Overlay)
//Storyboard Sound Samples
```

MVP 不生成 storyboard、break period 或自定义 hitsound sample。

### 14.9 `[TimingPoints]` 要求

第一版只支持单 BPM，因此只生成一个非继承 timing point。

格式：

```text
time,beatLength,meter,sampleSet,sampleIndex,volume,uninherited,effects
```

MVP 输出：

```text
offset,beatLength,4,1,0,100,1,0
```

其中：

```text
beatLength = 60000 / BPM
```

字段要求：

- `time` 为 offset，单位 ms；
- `beatLength` 为一拍长度，单位 ms；
- `meter` 第一版固定为 `4`；
- `sampleSet` 默认 `1`，即 Normal；
- `sampleIndex` 默认 `0`；
- `volume` 默认 `100`；
- `uninherited` 必须为 `1`；
- `effects` 默认 `0`。

可选：如果后续需要 kiai，可额外生成继承 timing point，但 MVP 不做。

### 14.10 `[Colours]` 要求

`[Colours]` 可选。

MVP 可不写该章节。若写入，可使用简单默认值：

```text
[Colours]
Combo1 : 255,192,0
Combo2 : 0,202,0
Combo3 : 18,124,255
Combo4 : 242,24,57
```

注意：上传的 `v128` 样例中颜色值包含 alpha，例如 `255,192,0,255`。MVP 使用 `v14` 时建议采用三通道 RGB，避免兼容性差异。

### 14.11 `[HitObjects]` 总体要求

`[HitObjects]` 中每一行代表一个物件。

osu!mania 第一版只生成两类物件：

- 普通 note；
- LN / hold note。

所有 HitObjects 必须按时间升序排列；同一时间的对象建议按 lane 升序排列。

### 14.12 4K 轨道坐标映射

osu!mania 使用 x 坐标映射列。

上传样例存在两种常见写法：

```text
64,192,320,448
```

以及：

```text
0,128,256,384
```

两者都可被 osu! 按列解析。为了稳定和直观，MVP 固定采用列中心坐标：

```text
Column 1: 64
Column 2: 192
Column 3: 320
Column 4: 448
```

`y` 坐标固定为：

```text
192
```

### 14.13 普通 note 格式

普通 note 格式：

```text
x,192,time,1,hitSound,hitSample
```

MVP 默认：

```text
x,192,time,1,0,0:0:0:0:
```

示例：

```text
64,192,3770,1,0,0:0:0:0:
```

字段要求：

- `x` 必须为对应轨道坐标；
- `time` 为整数毫秒；
- `type` 为 `1`；
- `hitSound` 默认 `0`；
- `hitSample` 默认 `0:0:0:0:`。

### 14.14 LN / Hold Note 格式

LN 格式：

```text
x,192,start_time,128,hitSound,end_time:hitSample
```

MVP 默认：

```text
x,192,start_time,128,0,end_time:0:0:0:0:
```

示例：

```text
320,192,1233,128,0,1490:0:0:0:0:
```

字段要求：

- `type` 必须为 `128`；
- `start_time` 和 `end_time` 都必须为整数毫秒；
- `end_time` 必须大于 `start_time`；
- LN 的 `end_time` 写在第六字段开头；
- 同一轨道 LN 区间内不得存在其他普通 note 或 LN。

### 14.15 Chord 写法

Chord 不需要特殊格式。

同一时间点在多个轨道写入多个 HitObject 即为 chord。

示例：

```text
64,192,1000,1,0,0:0:0:0:
192,192,1000,1,0,0:0:0:0:
```

要求：

- 同一时间同一轨道不能重复；
- 同一时间不同轨道允许共存；
- chord size 必须满足用户配置。

### 14.16 文件格式校验要求

导出前必须校验：

1. 第一行存在且为 `osu file format v14`；
2. `[General]` 中 `Mode: 3`；
3. `[Difficulty]` 中 `CircleSize:4`；
4. `[TimingPoints]` 至少一个非继承 timing point；
5. `[HitObjects]` 不为空；
6. HitObjects 时间升序；
7. 所有 note 时间为整数毫秒；
8. 所有 x 坐标属于 4K 坐标集合；
9. 所有 LN 的 end time 大于 start time；
10. 同一轨道不存在 LN 与 note 冲突；
11. 输出 zip 中存在 `AudioFilename` 对应的音频文件；
12. 如果写入背景图，zip 中必须存在对应文件。

---

## 15. 数据结构设计

### 15.1 DifficultyConfig

```python
@dataclass
class DifficultyConfig:
    version: str
    target_star: float | None
    target_msd: float | None
    chart_type: Literal["rice", "ln", "hybrid", "vibro"]
    key_style: Literal["jack", "tech", "speed", "stream"] | None
    allowed_subdivisions: list[str]
    chord_enabled: bool
    max_chord_size: int
    chord_probability: float
    max_jack_length: int
    max_anchor_length: int
    hand_balance: float
    ln_ratio: float
    min_ln_ms: int
    max_ln_ms: int
    hybrid_weights: dict[str, float]
    vibro_options: dict[str, Any]
```

### 15.2 NoteObject

```python
@dataclass
class NoteObject:
    time_ms: int
    lane: int
    end_time_ms: int | None = None

    @property
    def is_ln(self) -> bool:
        return self.end_time_ms is not None
```

### 15.3 AudioAnalysisResult

```python
@dataclass
class AudioAnalysisResult:
    bpm: float
    offset_ms: int
    duration_ms: int
    onset_times_ms: list[int]
    energy_curve: list[float]
    silent_regions: list[tuple[int, int]]
    beat_grid_ms: list[int]
    snap_points_ms: list[int]
```

---

## 16. 模块划分

### 16.1 UI 模块

职责：

- 收集用户输入；
- 展示分析结果；
- 展示生成日志；
- 提供下载按钮。

### 16.2 AudioAnalyzer

职责：

- 读取音频；
- 计算 BPM；
- 计算 offset；
- 检测 onset；
- 检测静音段；
- 可选运行 Demucs。

### 16.3 GridBuilder

职责：

- 根据 BPM 和 offset 建立 beat grid；
- 根据允许分辨率生成 snap points；
- 将候选 onset 吸附到 snap points。

### 16.4 PatternGenerator

职责：

- 生成 jack；
- 生成 tech；
- 生成 speed；
- 生成 stream；
- 生成 hybrid pattern；
- 生成 vibro pattern。

### 16.5 NoteGenerator

职责：

- 根据谱面类型生成 ordinary note 和 LN；
- 处理 chord；
- 控制局部密度。

### 16.6 Validator

职责：

- 检查 LN 冲突；
- 检查重叠 note；
- 检查最大 jack；
- 检查最大纵连；
- 检查静音段；
- 检查 chord size；
- 修复非法对象。

### 16.7 DifficultyEstimator

职责：

- 计算内部难度特征；
- 估计目标星数 / MSD 偏差；
- 给生成器反馈调参建议。

### 16.8 OsuExporter

职责：

- 生成 `.osu` 文本；
- 写入 metadata；
- 写入 timing points；
- 写入 hit objects。

### 16.9 Packager

职责：

- 复制音频；
- 复制背景图；
- 写入多个 `.osu` 文件；
- 打包 zip。

---

## 17. 一周开发计划

假设个人开发，每天 4 小时，总计约 28 小时。

### Day 1：项目框架与 `.osu` 导出

目标：能生成最小合法 4K `.osu`。

任务：

- 搭建 Streamlit 项目；
- 实现音频上传；
- 实现 metadata 表单；
- 实现 `.osu` 文件模板；
- 实现 4K x 坐标映射；
- 实现普通 note 写入；
- 实现 zip 打包下载。

验收：

- 用户上传音频后能下载 zip；
- zip 内含音频和 `.osu`；
- 修改后缀为 `.osz` 可导入 osu!。

### Day 2：音频分析与 beat grid

目标：能自动检测 BPM、offset、onset 和静音段。

任务：

- 使用 librosa 读取音频；
- 自动估计 BPM；
- 自动估计 onset；
- 实现 RMS 静音段检测；
- UI 支持手动 BPM / offset；
- 实现 beat grid；
- 实现分辨率 snap points。

验收：

- 能生成基于音频 onset 的候选 note 时间；
- 静音段不会生成普通 note。

### Day 3：Rice 生成与键型生成

目标：实现 rice 模式下 jack / tech / speed / stream。

任务：

- 实现 PatternGenerator；
- 实现 jack；
- 实现 stream；
- 实现 speed 受约束随机；
- 实现 tech 基础 pattern；
- 实现 chord size 限制；
- 实现最大 jack 限制。

验收：

- rice 模式可生成四种键型；
- 无同列重复爆炸；
- 可导出可玩谱。

### Day 4：LN / Hybrid / Vibro

目标：实现三类特殊模式。

任务：

- 实现 LN 对象；
- 实现 LN 合法性检查；
- 实现 LN 模式；
- 实现 hybrid 权重；
- 实现 vibro 模式；
- vibro 禁止 LN；
- rice 禁止 LN。

验收：

- LN 不与同列 note 冲突；
- hybrid 能按权重混合键型；
- vibro 能生成高频练习段。

### Day 5：质量检查与修复

目标：减少不可玩谱面。

任务：

- 实现 Validator；
- 检查同一时间同一轨重复；
- 检查 LN 重叠；
- 检查 LN 内 note；
- 检查最大纵连；
- 检查 chord size；
- 检查静音段；
- 实现修复策略。

验收：

- 输出谱面不含基本格式错误；
- 不出现同列 LN + note；
- 不出现超出限制的 jack。

### Day 6：难度估计与多难度生成

目标：支持用户生成多个难度。

任务：

- 实现内部难度特征计算；
- 实现目标星数 / MSD 输入；
- 实现简单迭代校正；
- 实现多个 DifficultyConfig；
- 每个难度单独生成 `.osu`。

验收：

- 用户可生成多个难度；
- 不同目标难度的 note 密度和 pattern 复杂度有明显区别。

### Day 7：UI 整理、打包与测试

目标：完成可演示版本。

任务：

- 整理 UI；
- 添加参数默认值；
- 添加生成日志；
- 添加错误提示；
- 测试不同音频；
- 修复 `.osu` 格式问题；
- 输出使用说明。

验收：

- 非开发者能通过 UI 上传音频并生成 `.osz`；
- 至少 3 首不同风格歌曲能生成可导入谱面；
- 生成谱面没有明显格式错误。

---

## 18. MVP 默认参数建议

```text
Key count: 4
BPM: 自动检测，允许手动修改
Offset: 自动检测，允许手动修改
Allowed subdivisions: 1/4, 1/8, 1/16, 1/6, 1/12
Max chord size: 2
Max jack length: 4
Hand balance: 0%
Min note interval: 40ms
Silent threshold: 自动估计
Rice LN ratio: 0
Vibro LN ratio: 0
LN min length: 250ms
LN max length: 2000ms
Difficulty correction loops: 3
```

---

## 19. 风险与应对

### 19.1 BPM 检测不准

风险：BPM 错误会导致全谱错位。

应对：

- UI 必须允许手动修改 BPM；
- UI 必须允许手动修改 offset；
- 可显示检测结果置信度。

### 19.2 生成谱面难度不准

风险：用户输入星数 / MSD，但生成结果偏差较大。

应对：

- MVP 中明确为近似控制；
- 使用内部特征估计；
- 通过迭代校正缩小偏差；
- 后续再接入真实难度算法。

### 19.3 谱面不可玩

风险：规则不足导致输出反人类。

应对：

- 强制 Validator；
- 限制 jack、chord、LN 冲突；
- 默认保守参数；
- vibro 作为特殊模式单独处理。

### 19.4 音轨分离过慢

风险：Demucs 影响用户体验。

应对：

- 默认关闭；
- 作为高级选项；
- 先用能量检测完成静音判断。


## 20. 验收标准

### 20.1 功能验收

项目完成时应满足：

- 用户可上传音频；
- 用户可选背景图；
- 用户可配置 metadata；
- 用户可生成至少一个 4K `.osu`；
- 用户可生成多个难度；
- 用户可选择 rice / LN / hybrid / vibro；
- 用户可选择 jack / tech / speed / stream；
- hybrid 可配置键型比例；
- 输出 zip 可改名为 `.osz` 并导入 osu!；
- 谱面不包含同轨 LN + note 冲突；
- 静音段不生成普通 note。

### 20.2 质量验收

至少使用 3 首不同风格歌曲测试：

1. 高 BPM 电子乐；
2. 普通流行歌；
3. 有明显静音 / breakdown 的歌曲。

每首至少生成：

- rice；
- LN 或 hybrid；
- vibro。

生成结果应满足：

- 能导入；
- 能播放；
- 能正常游玩；
- 没有明显同列冲突；
- 静音段没有大量 note；
- 难度变化可感知。

---

## 21. 后续版本方向

### 21.1 多 Key 支持

架构预留：

- key count 参数；
- lane x 坐标生成；
- 不同 key count 的 pattern 模板；
- 不同 key count 的难度估计。

### 21.2 难度算法接入

后续可接入或复刻：

- osu!mania star rating；
- Etterna MSD；
- 自定义练习难度指标。

### 21.3 编辑器预览

后续可增加：

- 时间轴；
- 波形显示；
- note 简易预览；
- 局部重生成；
- 参数调节后重新生成指定区间。

### 21.4 模型化生成

长期可探索：

- 基于规则生成的数据训练小模型；
- pattern 分类模型；
- 难度预测模型；
- 基于用户反馈的参数推荐。

---

## 22. 最终建议

第一版必须控制野心。

最重要的不是生成“像谱师写的谱”，而是完成以下闭环：

```text
上传音频
  ↓
配置练习目标
  ↓
生成合法 4K 谱面
  ↓
打包为 .osz
  ↓
导入 osu! 后可以玩
```

只要这个闭环稳定，后续才有价值继续优化难度算法、音轨分离、pattern 质量和多 key 支持。


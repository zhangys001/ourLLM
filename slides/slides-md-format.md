# slides 终端幻灯片 Markdown 编写规范

> 工具：maaslalani/slides (Go 编写)
> 来源：GitHub 官方 README + 示例文件

---

## 元信息头 (YAML Front Matter)

文件开头可选 YAML 块，用 `---` 包裹：

```yaml
---
theme: ./path/to/theme.json    # glamour 主题路径或远程 URL
author: Gopher                 # 左下角作者名（默认 OS 用户名，空字符串隐藏）
date: MMMM dd, YYYY           # 日期格式（默认 YYYY-MM-DD）
paging: "Slide %d / %d"       # 页码模板（默认 "Slide %d / %d"）
---
```

### 日期格式

| 格式 | 示例      |
|------|-----------|
| YYYY | 2006      |
| YY   | 06        |
| MMMM | January   |
| MMM  | Jan       |
| MM   | 01        |
| mm   | 1         |
| DD   | 02        |
| dd   | 2         |

---

## 幻灯片分隔

用 `---` 独占一行分隔幻灯片：

```markdown
# 第一页
内容

---

# 第二页
其他内容
```

---

## 支持的 Markdown 元素

| 元素         | 语法                       | 说明 |
|--------------|----------------------------|------|
| 标题 H1~H6  | `#` ~ `######`             | 完整支持 |
| 粗体         | `**text**`                 | |
| 斜体         | `*text*`                   | |
| 行内代码     | `` `code` ``               | |
| 无序列表     | `* item` / `- item`        | 支持嵌套 |
| 有序列表     | `1. item`                  | 支持嵌套 |
| 表格         | GFM 标准表格               | |
| 引用块       | `> text`                   | |
| 链接         | `[text](url)`              | |
| 图片         | `![alt](url)`              | 依赖终端能力 |
| Emoji        | `:smile:` 等               | |
| Graphviz     | `digraph { }`              | 需手动写 ASCII 渲染 |
| 注释         | `<!-- comment -->`         | |

### 注意

- `---` 是页面分隔符，正文中不可作水平线
- 图片在 iTerm2/Kitty 等现代终端支持内联渲染，老旧终端不显示

---

## 代码块

````markdown
```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, world!")
}
```
````

### 可执行代码

展示时按 `Ctrl+e` 执行代码块，输出显示在幻灯片末尾。

**支持语言**：bash, zsh, fish, elixir, go, javascript, python, ruby, perl, rust, java, cpp, swift, dart, v, lua, julia, scala

### 隐藏样板代码

用 `///` 前缀标记隐藏行。显示时忽略，执行和复制时保留：

````go
///package main
///
import "fmt"
///
///func main() {
fmt.Println("Hello!")
///}
````

---

## 预处理块

用三个波浪线 `~~~` + 命令名。展示前运行命令，stdout 替换块内容。

````markdown
```
~~~graph-easy --as=boxart
[ A ] - to -> [ B ]
~~~
```
````

输出变为 ASCII 图形：

```
┌───┐  to   ┌───┐
│ A │ ────> │ B │
└───┘       └───┘
```

**适用工具**：graph-easy（ASCII 图）、plantuml（UML）、xargs cat（导入文件）等。

**导入外部文件**：

```markdown
~~~xargs cat
path/to/file.md
~~~
```

---

## 内建主题

| 主题   | 说明                               |
|--------|------------------------------------|
| 默认   | 使用 Unicode 框线渲染              |
| `ascii` | 纯 ASCII 字符（YAML 头指定 `theme: ascii`） |

自定义主题：JSON 文件，符合 glamour 主题格式。

---

## 重要限制

1. **无逐步显示 (fragments)**：每页内容一次性全出
2. **无多列布局**：纯 markdown 流式渲染
3. **不支持脚注、任务列表、删除线**等扩展 GFM
4. **预处理需执行权限**：`chmod +x file.md`
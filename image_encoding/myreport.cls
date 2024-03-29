% \NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{myreport}
              [2017/09/16 v2.6 Standard LaTeX Template for CUMCM]
%% Inital Code

%% Loading Class
\LoadClass[a4paper,12pt]{article}

%% Declaration of Options

% 需要单独封面的话,使用参数 pdfCover
% 并将封面放到 figures/cover.pdf
\DeclareOption{pdfCover}{
	\renewcommand{\maketitle}{
  	\includepdf[pages={1}]{figures/cover.pdf}
	}
}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}


%% Executing of Options
\ProcessOptions\relax

%% Loading Package
\RequirePackage{ifxetex}
\RequireXeTeX
\ifxetex\else
\ClassError{myreportthesis}{You must use the `xelatex' driver\MessageBreak Please choose `xelatex'}{%
Just choose `xelatex', no `pdflatex' or `latex' and so on.}
\fi

% 支持中文的 ctex 宏包
\RequirePackage{ctex}
% 页面布局
\RequirePackage{geometry}
% 数学宏包
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{amssymb}
\RequirePackage{bm}
\RequirePackage{mathtools}
%\RequirePackage{titletoc}
% 设置颜色
\RequirePackage{xcolor}
% 插入图片
\RequirePackage{graphicx}
\RequirePackage{float}
% 表格
\RequirePackage{array}
%% 长表格
\RequirePackage{longtable}
%% booktabs 提供了\toprule 等命令.
\RequirePackage{booktabs}
%% multirow 支持在表格中跨行
\RequirePackage{multirow}
%% 调整间隔, 让表格更好看些
\RequirePackage{bigstrut}
%% 在跨行表格中输入定界符
\RequirePackage{bigdelim}
% 保护脆落命令
\RequirePackage{cprotect}
% 设置代码环境
\RequirePackage{listings}
\RequirePackage{xcolor}
% 插入链接
\RequirePackage{url}
% 绘图的包
\RequirePackage{tikz}

% 字图与子标题
% \RequirePackage{subcaption}
\RequirePackage[titles]{tocloft}
\RequirePackage{subfig}
% usage https://tex.stackexchange.com/questions/111822/using-subfloat-to-place-figures-side-by-side

% 物理单位
\RequirePackage{siunitx}

% 表格
\RequirePackage{tabularx}
\RequirePackage{tabu}


\sisetup{
  list-final-separator = { ~\text{和}~ },  % usage \SIlist{1;2;3}{m}
  list-pair-separator = { ~\text{和}~ },
  range-phrase = { \sim }, % usage \SIrange{1}{2}{m}
}

% 插入PDF
\RequirePackage{pdfpages}
%\renewcommand{\cftdot}{$\cdot$}
%\renewcommand{\cftsecdotsep}{1.5}
\renewcommand{\cftsecdotsep}{4.5}
\setlength{\cftbeforesecskip}{7pt}
\setlength{\cftbeforesubsecskip}{3pt}
%\setlength{\cftbeforesubsecskip}{5pt}
\renewcommand{\cftsecfont}{\bfseries\zihao{-4}\heiti}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
\renewcommand{\cftsecaftersnumb}{\hskip.4em}



\definecolor{dkgreen}{rgb}{0,0.6,0}
\definecolor{gray}{rgb}{0.5,0.5,0.5}
\definecolor{mauve}{rgb}{0.58,0,0.82}

\lstset{
  frame=tb,
  aboveskip=3mm,
  belowskip=3mm,
  showstringspaces=false,
  columns=flexible,
  framerule=1pt,
  rulecolor=\color{gray!35},
  backgroundcolor=\color{gray!5},
  basicstyle={\small\ttfamily},
  numbers=left,
  numberstyle=\tiny\color{gray},
  keywordstyle=\color{blue},
  commentstyle=\color{dkgreen},
  stringstyle=\color{mauve},
  breaklines=true,
  breakatwhitespace=true,
  tabsize=3,
}
% 首行缩进
\RequirePackage{indentfirst}
% 设置浮动体的标题
\RequirePackage{caption}
% 定制列表环境
\RequirePackage{enumitem}
% 下划线
\RequirePackage{ulem}
% 尺寸计算
\RequirePackage{calc}

\RequirePackage[titletoc,title]{appendix}

%\RequirePackage[title,toc,titletoc]{appendix}

\RequirePackage{etoolbox}
\AtBeginEnvironment{thebibliography}{%
    \phantomsection
    \addcontentsline{toc}{section}{\refname}
    }
% 超链接 hyperref 的设置
\RequirePackage[pdfusetitle]{hyperref}
% 智能自动化交叉引用
\RequirePackage{cleveref}
%% Document Layout
% 页面布局
\geometry{top=25mm,bottom=25mm,left=25mm,right=25mm}
\renewcommand*{\baselinestretch}{1.38}
% 修改tabular 环境, 设置表格中的行间距为正文行间距.
\let\myreport@oldtabular\tabular
\let\myreport@endoldtabular\endtabular
\renewenvironment{tabular}%
	{\bgroup%
	\renewcommand{\arraystretch}{1.38}%
	\myreport@oldtabular}%
	{\myreport@endoldtabular\egroup}
% 每行缩进两个汉字
\setlength\parindent{2em}
% 设置字体
\setmainfont{Times New Roman}
%\setmonofont{Courier New}
\setsansfont{Arial}
\setCJKfamilyfont{kai}[AutoFakeBold]{simkai.ttf}
\newcommand*{\kai}{\CJKfamily{kai}}
\setCJKfamilyfont{song}[AutoFakeBold]{SimSun}
\newcommand*{\song}{\CJKfamily{song}}

\renewcommand\normalsize{%
	\@setfontsize\normalsize{12.05}{14.45}%
	\abovedisplayskip 12\p@ \@plus3\p@ \@minus7\p@
	\abovedisplayshortskip \z@ \@plus3\p@
	\belowdisplayshortskip 6.5\p@ \@plus3.5\p@ \@minus3\p@}
% 数学环境, 定理等设置
\newtheorem{definition}{\myreport@cap@definition}
\newtheorem{theorem}{\myreport@cap@theorem}
\newtheorem{lemma}{\myreport@cap@lemma}
\newtheorem{corollary}{\myreport@cap@corollary}
\newtheorem{assumption}{\myreport@cap@assumption}
\newtheorem{conjecture}{\myreport@cap@conjecture}
\newtheorem{axiom}{\myreport@cap@axiom}
\newtheorem{principle}{\myreport@cap@principle}
\newtheorem{problem}{\myreport@cap@problem}
\newtheorem{example}{\myreport@cap@example}
\newtheorem{proof}{\myreport@cap@proof}
\newtheorem{solution}{\myreport@cap@solution}

% 浮动环境设置
% 默认情况下, \LaTeX{} 要求每页的文字至少占据 20%,否则该页就只单独放置一个浮动环境,
% 而这通常不是我们想要的, 我们将这个要求降低到 5%.
 \renewcommand*{\textfraction}{0.05}
% 有时如果多个浮动环境连续放在一起, \LaTeX{}
% 会将它们分在几个不同页，即使它们可在同一页放
% 得下. 我们可以通过修改 |\topfraction| 和 |\bottomfraction| 分别设置顶端和底端的浮
% 动环境的最大比例.
 \renewcommand*{\topfraction}{0.9}
 \renewcommand*{\bottomfraction}{0.8}
% 有时\LaTeX{}会把一个浮动环境单独放在一页,
% 我们要求这个环境至少要占据 85% 才能单独放在一页.
% 注意:  |\floatpagefraction| 的数值必须小于 |\topfraction|.
 \renewcommand*{\floatpagefraction}{0.85}
% 关于图片 graphicx
 % 如果图片没有指定后缀, 依次按下列顺序搜索
\DeclareGraphicsExtensions{.pdf,.eps,.jpg,.png}
 % 设置图表搜索路径
\graphicspath{{figures/}}
% 图表标题
\DeclareCaptionFont{song}{\songti}
\DeclareCaptionFont{minusfour}{\zihao{-4}}
\captionsetup[figure]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour,bf}, % 图的字体, 宋体小四
	position=bottom % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}
\captionsetup[table]{%
	format=hang,   % 标题从第二行开始都有缩进, 应该和 justification=raggedright 的效果一样.
	labelsep=quad, % 分隔符是一个空格
	font={song,minusfour,bf}, % 表的字体, 宋体小四
	position=top % position=bottom, 不代表标题放在下面, 标题仍放在你放\caption的位置.
}
% 列表环境设置
\setlist{%
	topsep=0.3em, % 列表顶端的垂直空白
	partopsep=0pt, % 列表环境前面紧接着一个空白行时其顶端的额外垂直空白
	itemsep=0ex plus 0.1ex, % 列表项之间的额外垂直空白
	parsep=0pt, % 列表项内的段落之间的垂直空白
	leftmargin=1.5em, % 环境的左边界和列表之间的水平距离
	rightmargin=0em, % 环境的右边界和列表之间的水平距离
	labelsep=0.5em, % 包含标签的盒子与列表项的第一行文本之间的间隔
	labelwidth=2em % 包含标签的盒子的正常宽度；若实际宽度更宽，则使用实际宽度。
}

% 超链接
	 \hypersetup{%
%         xetex,
		% unicode=false, % hyperref 和 xetex 同时使用时不能开启 Unicode 选项.
         pdfstartview=FitH,
         CJKbookmarks=true,
         bookmarksnumbered=true,
         bookmarksopen=true,
         colorlinks, %注释掉此项则交叉引用为彩色边框(将colorlinks和pdfborder同时注释掉)
         pdfborder=001,   %注释掉此项则交叉引用为彩色边框
		 allcolors=black,
		 breaklinks=true}%
% \if@myreport@bwprint
% \AtBeginDocument{\hypersetup{hidelinks}}
% \else\relax\fi
\pdfstringdefDisableCommands{%
	\def\cftdotfill{ }%
}
\lstnewenvironment{tcode}
  {
    \lstset{basicstyle = \small\ttfamily,
        language=TeX,
        tabsize = 4,
        frame = single,
        escapechar = `,
        breaklines = true,
        breakatwhitespace = true,
        frameround = tttt,
    }
  }
  {}
%%% crefformat settings
\crefformat{figure}{#2图~#1#3}
\crefrangeformat{figure}{图~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{figure}{图~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{table}{#2表#1#3}
\crefrangeformat{table}{表(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{table}{表~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{equation}{#2公式(#1#3)}
\crefrangeformat{equation}{公式(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{equation}{公式(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{definition}{#2\myreport@cap@definition~#1#3}
\crefrangeformat{definition}{\myreport@cap@definition~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{definition}{\myreport@cap@definition~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{theorem}{#2\myreport@cap@theorem~#1#3}
\crefrangeformat{theorem}{\myreport@cap@theorem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{theorem}{\myreport@cap@theorem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{lemma}{#2\myreport@cap@lemma~#1#3}
\crefrangeformat{lemma}{\myreport@cap@lemma~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{lemma}{\myreport@cap@lemma~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{corollary}{#2\myreport@cap@corollary~#1#3}
\crefrangeformat{corollary}{\myreport@cap@corollary~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{corollary}{\myreport@cap@corollary~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{assumption}{#2\myreport@cap@assumption~#1#3}
\crefrangeformat{assumption}{\myreport@cap@assumption~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{assumption}{\myreport@cap@assumption~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{conjecture}{#2\myreport@cap@conjecture~#1#3}
\crefrangeformat{conjecture}{\myreport@cap@conjecture~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{conjecture}{\myreport@cap@conjecture~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{axiom}{#2\myreport@cap@axiom~#1#3}
\crefrangeformat{axiom}{\myreport@cap@axiom~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{axiom}{\myreport@cap@axiom~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{principle}{#2\myreport@cap@principle~#1#3}
\crefrangeformat{principle}{\myreport@cap@principle~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{principle}{\myreport@cap@principle~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{problem}{#2\myreport@cap@problem~#1#3}
\crefrangeformat{problem}{\myreport@cap@problem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{problem}{\myreport@cap@problem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{example}{#2\myreport@cap@example~#1#3}
\crefrangeformat{example}{\myreport@cap@example~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{example}{\myreport@cap@example~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{proof}{#2\myreport@cap@proof~#1#3}
\crefrangeformat{proof}{\myreport@cap@proof~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{proof}{\myreport@cap@proof~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{solution}{#2\myreport@cap@solution~#1#3}
\crefrangeformat{solution}{\myreport@cap@solution~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{solution}{\myreport@cap@solution~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 中文标题格式设置
% 目录深度
\setcounter{secnumdepth}{3}
% 节标题格式, 居中, 使用\chinese命令修改计数器
\def\@seccntformat#1{\csname the#1\endcsname\ }

% 一、section
\newcommand{\chineseCenterSection}{
  \renewcommand\thesection{\chinese{section}、}
  \renewcommand\section{\@startsection{section}{1}{\z@}%
  {-3.5ex \@plus -1ex \@minus -.2ex}%
  {2.3ex \@plus.2ex}%
  {\centering\normalfont\Large\bfseries}}

  % 1. 1 subsection
  \renewcommand\thesubsection{\arabic{section}.\arabic{subsection}}
  \renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  	{-3.25ex\@plus -1ex \@minus -.2ex}%
  	{1.5ex \@plus .2ex}%
  	{\normalfont\large\bfseries}}
}

% (一) subsection
\newcommand{\chineseSubsection}{
  \renewcommand\thesubsection{(\chinese{subsection})\thinspace}
  \renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\large\bfseries}}
  % 1. 1 .1 subsubsection
  \renewcommand\thesubsubsection{%
    \arabic{section}.\arabic{subsection}.\arabic{subsubsection}%
  }
  \renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
}


% 1. subsubsection
\newcommand{\singleArabicSubsubsection}{
  \renewcommand\thesubsubsection{\arabic{subsubsection}.}
  \renewcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
  {-3.25ex\@plus -1ex \@minus -.2ex}%
  {1.5ex \@plus .2ex}%
  {\normalfont\normalsize\bfseries}}
}


% \renewcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
% 	{3.25ex \@plus1ex \@minus.2ex}%
% 	{-1em}%
% 	{\normalfont\normalsize\bfseries}}
% \renewcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
% 	 {3.25ex \@plus1ex \@minus .2ex}%
% 	 {-1em}%
% 	 {\normalfont\normalsize\bfseries}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define new environment and command
%% 摘要两个字设置为 4 号.
%% 定义摘要环境
\renewenvironment{abstract}{%
\if@twocolumn
		\section*{\abstractname}%
	\else
	 \begin{center}%
	{\zihao{4}\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
	 \end{center}%
 	 \quotation
	\fi}
	{\if@twocolumn\else\endquotation\newpage\null\fi}
\renewenvironment{quotation}
	{\list{}{\listparindent 2em%
	 \itemindent \listparindent
	 \rightmargin\z@
	 \leftmargin\z@
	 \parsep \z@ \@plus\p@}%
	\item\relax}
	{\endlist}
%
\newcommand\keywords[1]{%
	\renewcommand{\myreport@tokens@keywords}{#1}
	\par
	\vskip1ex
	{\noindent\zihao{-4}\heiti\myreport@cap@keywordsname：}~{\myreport@tokens@keywords}
	}
% 定义附录

% \renewcommand{\setthesection}{\appendixname\Alph{section}}
% \renewcommand\appendix{\par
	% \setcounter{section}{0}%
	 % \setcounter{subsection}{0}%
	% \gdef\thesection{\appendixname\@Alph\c@section}}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initialization
 %%%数学定理相关的常量
\newcommand*{\myreport@cap@definition}{定义}
\newcommand*{\myreport@cap@theorem}{定理}
\newcommand*{\myreport@cap@lemma}{引理}
\newcommand*{\myreport@cap@corollary}{推论}
\newcommand*{\myreport@cap@assumption}{假设}
\newcommand*{\myreport@cap@conjecture}{猜想}
\newcommand*{\myreport@cap@axiom}{公理}
\newcommand*{\myreport@cap@principle}{定律}
\newcommand*{\myreport@cap@problem}{问题}
\newcommand*{\myreport@cap@example}{例}
\newcommand*{\myreport@cap@proof}{证明}
\newcommand*{\myreport@cap@solution}{解}
 %%%  中文标题名称
\newcommand*{\myreport@cap@contentsname}{目录}
\newcommand*{\myreport@cap@listfigurename}{插图清单}
\newcommand*{\myreport@cap@listtablename}{附表清单}
\newcommand*{\myreport@cap@refname}{参考文献}
\newcommand*{\myreport@cap@indexname}{索引}
\newcommand*{\myreport@cap@figurename}{图}
\newcommand*{\myreport@cap@tablename}{表}
\newcommand*{\myreport@cap@appendixname}{附录}
\newcommand*{\myreport@cap@abstractname}{摘要}
 %%%  中文关键字
\newcommand*{\myreport@cap@keywordsname}{关键字}


%设置目录格式
% 节的目录格式
% \titlecontents{section}[0pt]{\vspace{2mm}\bfseries}
% {\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}
% % 小节的目录格式
% \titlecontents{subsection}[30pt]{\songti}
% {\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}
% \titlecontents{subsubsection}[55pt]{\songti}
% {\thecontentslabel\hskip.5em}{}{\titlerule*[0.5pc]{.}\contentspage}

% 上标文献
\newcommand{\upcite}[1]{$^{\mbox{\scriptsize \cite{#1}}}$}
\newcommand{\supercite}[1]{\textsuperscript{\textsuperscript{\cite{#1}}}}


% 修改input路径
\makeatletter
\providecommand*\input@path{}
\newcommand\addinputpath[1]{
\expandafter\def\expandafter\input@path
\expandafter{\input@path{#1}}}
\addinputpath{figures/}
\makeatother

% 微分命令
  %% 参考 https://liam.page/2017/05/01/the-correct-way-to-use-differential-operator/
\newcommand*{\dif}{\mathop{}\!\mathrm{d}}

% matlab2tikz 的相关内容
  %% 参考https://github.com/matlab2tikz/matlab2tikz
\RequirePackage{pgfplots}
\pgfplotsset{compat=newest}
%% the following commands are needed for some matlab2tikz features
\usetikzlibrary{plotmarks}
\usetikzlibrary{arrows.meta}
\usepgfplotslibrary{patchplots}
\RequirePackage{grffile}
\RequirePackage{amsmath}
%% you may also want the following commands
%\pgfplotsset{plot coordinates/math parser=false}
%\newlength\figureheight
%\newlength\figurewidth

% 设置参考文献格式
% \bibliographystyle{plain}
% \RequirePackage{gbt7714}

% 优化大于等于和小于等于
\renewcommand{\le}{\leqslant}
\renewcommand{\ge}{\geqslant}

% 算法环境
\RequirePackage{algorithm}
% \RequirePackage{clrscode}
\RequirePackage{clrscode3e}
\floatname{algorithm}{算法}
\renewcommand{\Comment}[1][2]{$\hspace*{#1em}\rhd$ }

% 页眉含有章节标题
\RequirePackage{fancyhdr}
\fancyhead{} % Clear the headers
\renewcommand{\headrulewidth}{0pt} % Width of line at top of page
\fancyhead[L]{\slshape\leftmark} % Mark right [R] of page with Chapter name [\leftmark]
\pagestyle{fancy} % Set default style for all content pages (not TOC, etc)

% % 防止目录重叠 https://www.latexstudio.net/archives/7986.html
% \makeatletter
% \renewcommand{\numberline}[1]{%
% \settowidth\@tempdimb{#1\hspace{0.15em}}%
% \ifdim\@tempdima<\@tempdimb%
% \@tempdima=\@tempdimb%
% \fi%
% \hb@xt@\@tempdima{\@cftbsnum #1\@cftasnum\hfil}\@cftasnumb}
% \makeatother


%% crefformat settings
\crefformat{figure}{#2图~#1#3}
\crefrangeformat{figure}{图~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{figure}{图~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{table}{#2表#1#3}
\crefrangeformat{table}{表(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{table}{表~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

\crefformat{equation}{#2公式(#1#3)}
\crefrangeformat{equation}{公式(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{equation}{公式(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{definition}{#2\mcm@cap@definition~#1#3}
\crefrangeformat{definition}{\mcm@cap@definition~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{definition}{\mcm@cap@definition~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{theorem}{#2\mcm@cap@theorem~#1#3}
\crefrangeformat{theorem}{\mcm@cap@theorem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{theorem}{\mcm@cap@theorem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{lemma}{#2\mcm@cap@lemma~#1#3}
\crefrangeformat{lemma}{\mcm@cap@lemma~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{lemma}{\mcm@cap@lemma~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{corollary}{#2\mcm@cap@corollary~#1#3}
\crefrangeformat{corollary}{\mcm@cap@corollary~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{corollary}{\mcm@cap@corollary~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{assumption}{#2\mcm@cap@assumption~#1#3}
\crefrangeformat{assumption}{\mcm@cap@assumption~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{assumption}{\mcm@cap@assumption~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{conjecture}{#2\mcm@cap@conjecture~#1#3}
\crefrangeformat{conjecture}{\mcm@cap@conjecture~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{conjecture}{\mcm@cap@conjecture~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{axiom}{#2\mcm@cap@axiom~#1#3}
\crefrangeformat{axiom}{\mcm@cap@axiom~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{axiom}{\mcm@cap@axiom~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{principle}{#2\mcm@cap@principle~#1#3}
\crefrangeformat{principle}{\mcm@cap@principle~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{principle}{\mcm@cap@principle~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{problem}{#2\mcm@cap@problem~#1#3}
\crefrangeformat{problem}{\mcm@cap@problem~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{problem}{\mcm@cap@problem~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{example}{#2\mcm@cap@example~#1#3}
\crefrangeformat{example}{\mcm@cap@example~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{example}{\mcm@cap@example~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{proof}{#2\mcm@cap@proof~#1#3}
\crefrangeformat{proof}{\mcm@cap@proof~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{proof}{\mcm@cap@proof~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{solution}{#2\mcm@cap@solution~#1#3}
\crefrangeformat{solution}{\mcm@cap@solution~(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{solution}{\mcm@cap@solution~(#2#1#3)}{ 和~(#2#1#3)}{，(#2#1#3)}{ 和~(#2#1#3)}

\crefformat{algorithm}{#2算法#1#3}
\crefrangeformat{algorithm}{算法(#3#1#4)\;\~{}\;(#5#2#6)}
\crefmultiformat{algorithm}{算法~(#2#1#3)}{和~(#2#1#3)}{，(#2#1#3)}{和~(#2#1#3)}

% <- 的定义
\DeclareMathSymbol{\leftarrow}{\mathrel}{symbols}{"20}
\let\gets=\leftarrow


% 符号表
\RequirePackage[symbols,nogroupskip,sort=none]{glossaries-extra}

% new keys must be defined before use
\glsaddstoragekey{unit}{}{\glossentryunit}
\glsnoexpandfields

\newglossarystyle{symbunitlong}{%
  \setglossarystyle{long3col}% base this style on the list style
  \renewenvironment{theglossary}{% Change the table type --> 3 columns
    \begin{longtable}{lp{\glsdescwidth}>{\centering\arraybackslash}p{2cm}}%
  }{
    \end{longtable}
  }%
  %
  \renewcommand*{\glossaryheader}{%  Change the table header
    \toprule[1.5pt]
    \bfseries Symbol & \bfseries Description & \bfseries Unit\\\hline
    \endhead
    \bottomrule[1.5pt]
    \vspace{2pt}
    \endfoot
  }%
  \renewcommand*{\glossentry}[2]{%  Change the displayed items
    \glstarget{##1}{\glossentryname{##1}} %
    & \glossentrydesc{##1}% Description
    & \glossentryunit{##1}  \tabularnewline
  }%
}

\newglossarystyle{symblong}{%
  \setglossarystyle{long}% base this style on the list style
  \renewenvironment{theglossary}{% Change the table type --> 3 columns
    \begin{longtable}{lp{0.9\textwidth}}%
    % \begin{longtable}{cc}%

    % \begin{longtable}
  }{
    \end{longtable}
  }%
  %
  \renewcommand*{\glossaryheader}{%  Change the table header
    \toprule[1.5pt]
    \bfseries 符号 & \bfseries 说明 \\
    \hline
    \endhead
    \bottomrule[1.5pt]
    \vspace{2pt}
    \endfoot
  }%
  \renewcommand*{\glossentry}[2]{%  Change the displayed items
    \glstarget{##1}{\glossentryname{##1}} %
    & \glossentrydesc{##1}% Description
    \tabularnewline
  }%
}
\renewcommand{\glossarysection}[2][]{}

% abstract
\RequirePackage[main=pinyin,english]{babel}
\addto\captionspinyin{
	\renewcommand\keywords[1]{%
		% \par
		\vskip1ex
		{\noindent\zihao{-4}\heiti 关键字：}~{#1}
	}
  \renewcommand{\refname}{参考文献}
  \renewcommand{\figurename}{图}
  \renewcommand{\tablename}{表}
  \renewcommand\appendixname{附录}
  \renewcommand{\contentsname}{目录}
  \renewcommand\appendixtocname{附录}
  \renewcommand{\abstractname}{摘要}
}
\addto\captionsenglish{
	\renewcommand\keywords[1]{%
		% \par
		\vskip1ex
		\textbf{Keywords}:  #1
	}
}

% \MyTitle
\RequirePackage{authoraftertitle}

% 文件树（|- 等tree命令生成符号）
\RequirePackage{ifplatform}
\ifwindows
\else
  \setmonofont{DejaVu Sans Mono}
\fi

\endinput

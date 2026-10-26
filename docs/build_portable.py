"""Build the public documentation as a portable HTML page or web assets."""
from pathlib import Path
import argparse
from html.parser import HTMLParser
import json
import posixpath
import re
import html
import markdown
from pygments.formatters import HtmlFormatter
from pymdownx.slugs import slugify as make_slugify
import yaml

ICONS={
 'search':'<path d="m21 21-5-5"/><circle cx="10.5" cy="10.5" r="7.5"/>',
 'menu':'<path d="M4 6h16M4 12h16M4 18h16"/>',
 'chevron-right':'<path d="m9 5 7 7-7 7"/>',
 'chevron-down':'<path d="m6 9 6 6 6-6"/>',
 'copy':'<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/>',
 'link':'<path d="m10 13 4-4m-6 5-2 2a4 4 0 0 1-6-6l4-4a4 4 0 0 1 6 0m2 4 2-2a4 4 0 0 1 6 6l-4 4a4 4 0 0 1-6 0" transform="translate(1 1)"/>',
 'print':'<path d="M6 8V3h12v5M6 17H3V8h18v9h-3"/><rect x="6" y="13" width="12" height="8"/>',
 'arrow-left':'<path d="M20 12H4m6-6-6 6 6 6"/>',
 'arrow-right':'<path d="M4 12h16m-6-6 6 6-6 6"/>',
 'arrow-up':'<path d="M12 20V4m-6 6 6-6 6 6"/>',
}
ICONS={key:f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{value}</svg>' for key,value in ICONS.items()}
GITHUB='<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .9a11.1 11.1 0 0 0-3.51 21.63c.55.1.76-.24.76-.54v-2.13c-3.1.68-3.75-1.32-3.75-1.32-.51-1.29-1.24-1.63-1.24-1.63-1.02-.7.08-.68.08-.68 1.12.08 1.71 1.15 1.71 1.15 1 .1.8 2.29 3.22 1.62.1-.71.39-1.2.7-1.48-2.48-.28-5.08-1.24-5.08-5.52 0-1.22.44-2.21 1.14-2.99-.12-.28-.5-1.42.11-2.95 0 0 .94-.3 3.05 1.14a10.68 10.68 0 0 1 5.55 0c2.12-1.43 3.05-1.14 3.05-1.14.61 1.53.23 2.67.11 2.95.71.78 1.14 1.77 1.14 2.99 0 4.29-2.61 5.23-5.1 5.51.4.35.76 1.02.76 2.06v3.42c0 .31.2.65.77.54A11.1 11.1 0 0 0 12 .9Z"/></svg>'

slugify=make_slugify(case='lower')

class PlainText(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]
    def handle_data(self,data): self.parts.append(data)

def diagram_svg(kind,language='zh'):
    # Accessible code-native diagrams for portable/offline reading.
    def node(x,y,w,label,sub=''):
        return f'<rect x="{x}" y="{y}" width="{w}" height="58" rx="6" fill="#fff" stroke="#d7ddec"/><text x="{x+w/2}" y="{y+25}" text-anchor="middle" font-size="12" fill="#4b5481">{html.escape(label)}</text><text x="{x+w/2}" y="{y+43}" text-anchor="middle" font-size="9" fill="#8991ae">{html.escape(sub)}</text>'
    def arrow(x,y,x2,y2): return f'<path d="M{x} {y} L{x2} {y2}" stroke="#a2accd" fill="none" stroke-width="1.4" marker-end="url(#head)"/>'
    if kind=='architecture':
        shapes=node(210,10,180,'FastAPI / LLMService','请求接入 · 副本路由')+arrow(300,68,300,96)+node(210,100,180,'Controller','配置节点控制面')
        shapes+=arrow(245,158,145,194)+arrow(355,158,455,194)
        shapes+=node(60,198,170,'Prefill → Decode','P/D 分离')+node(370,198,170,'Simple Engine','同一引擎执行两阶段')
        shapes+=arrow(145,256,145,285)+arrow(455,256,455,285)+node(60,289,170,'GPU Workers','模型前向 · 独立 GPU KV')+node(370,289,170,'GPU Workers','模型前向 · 独立 GPU KV')
        shapes+=node(205,382,190,'QuickCache / BlockManager','共享 CPU 权重与 KV')+arrow(145,347,235,380)+arrow(455,347,365,380)
        size='0 0 600 455'; label='请求路由、节点控制面与两种执行模式'
    else:
        shapes=node(8,25,96,'ABSENT','未部署')+arrow(104,54,127,54)+node(132,25,96,'LOADING','加载缓存')+arrow(228,54,253,54)+node(258,25,96,'READY','接受新请求')+arrow(354,54,378,54)+node(383,25,102,'DRAINING','排空与卸载')
        shapes+=arrow(180,83,180,135)+node(132,140,96,'FAILED','部署失败')
        shapes+='<path d="M434 84 V220 H55 V84" stroke="#a2accd" fill="none" stroke-width="1.4" marker-end="url(#head)"/>'
        shapes+='<text x="300" y="213" text-anchor="middle" font-size="10" fill="#8991ae">卸载成功</text>'
        size='0 0 500 240'; label='模型缓存副本的主要生命周期'
    result=f'<figure class="diagram"><svg viewBox="{size}" role="img" aria-label="{label}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="head" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 8 4 0 8" fill="#a2accd"/></marker></defs>{shapes}</svg><figcaption>{label}</figcaption></figure>'
    if language=='en':
        translations={'请求接入 · 副本路由':'API · Replica routing','配置节点控制面':'Configured node control plane','P/D 分离':'P/D separation','同一引擎执行两阶段':'Both stages in one engine','模型前向 · 独立 GPU KV':'Forward · Per-worker GPU KV','共享 CPU 权重与 KV':'Shared CPU weights and KV','请求路由、节点控制面与两种执行模式':'Request routing, node control plane, and execution modes','未部署':'Not deployed','加载缓存':'Loading cache','接受新请求':'Accepts new requests','排空与卸载':'Drain and unload','部署失败':'Deployment failed','卸载成功':'Unload succeeded','模型缓存副本的主要生命周期':'Cached model replica lifecycle'}
        for original,translated in translations.items(): result=result.replace(original,translated)
    return result

def build_language(root,language):
    doc=root/'docs'/language
    config=yaml.load((root/f'mkdocs.{language}.yml').read_text(encoding='utf-8'),Loader=yaml.BaseLoader)
    path_to_id={path.relative_to(doc).as_posix():('examples' if path.name=='README.md' else path.stem) for path in doc.rglob('*.md')}
    groups=[]; pages={}
    def leaves(items):
        result=[]
        for item in items:
            for title,value in item.items():
                if isinstance(value,list): result.extend(leaves(value))
                else: result.append((title,value))
        return result
    for index,item in enumerate(config['nav']):
        title,children=next(iter(item.items())); group_id=f'group-{index}'
        if isinstance(children,str): sections=[{'title':'','entries':[(title,children)]}]
        elif any(isinstance(next(iter(child.values())),list) for child in children):
            sections=[{'title':subtitle,'entries':leaves(subchildren)} for child in children for subtitle,subchildren in child.items()]
        else: sections=[{'title':'','entries':leaves(children)}]
        group={'id':group_id,'title':title,'sections':[]}
        for section in sections:
            ids=[]
            for label,path in section['entries']:
                page_id=path_to_id[path]; ids.append(page_id)
                text=(doc/path).read_text(encoding='utf-8')
                md=markdown.Markdown(extensions=['tables','fenced_code','sane_lists','toc','codehilite'],extension_configs={'toc':{'slugify':slugify,'permalink':'¶'},'codehilite':{'guess_lang':False,'css_class':'highlight','noclasses':False}})
                if path in ('architecture.md','model-management.md'):
                    kind='architecture' if path=='architecture.md' else 'lifecycle'
                    text=re.sub(r'```mermaid\s*\n.*?\n```',lambda _:diagram_svg(kind,language),text,flags=re.S)
                body=md.convert(text)
                def replace_link(match):
                    target=html.unescape(match.group(1))
                    if target.startswith(('https:','http:','mailto:')): return match.group(0)
                    if target.startswith('#'): return 'href="#/'+language+'/'+page_id+target+'"'
                    source,sep,anchor=target.partition('#')
                    normalized=posixpath.normpath(posixpath.join(posixpath.dirname(path),source))
                    if normalized in path_to_id: return 'href="#/'+language+'/'+path_to_id[normalized]+('#'+anchor if sep else '')+'"'
                    if normalized.startswith('examples/') and (doc/normalized).is_file():
                        # Files are available as embedded download links in the portable edition.
                        import base64
                        raw=(doc/normalized).read_bytes()
                        return f'href="data:application/octet-stream;base64,{base64.b64encode(raw).decode()}" download="{html.escape(Path(normalized).name)}"'
                    return match.group(0)
                body=re.sub(r'href="([^"]+)"',replace_link,body)
                headings=[]
                for token in md.toc_tokens:
                    for child in token.get('children',[]):
                        if child['level'] in (2,3):
                            headings.append({'id':child['id'],'title':child['name'],'level':child['level']})
                            for sub in child.get('children',[]):
                                if sub['level']==3: headings.append({'id':sub['id'],'title':sub['name'],'level':3})
                parser=PlainText(); parser.feed(body)
                pages[page_id]={'id':page_id,'label':label,'group':group_id,'html':body,'headings':headings,'text':re.sub(r'\s+',' ',' '.join(parser.parts))}
            group['sections'].append({'title':section['title'],'pages':ids})
        group['first']=group['sections'][0]['pages'][0]; groups.append(group)
    return {'pages':pages,'groups':groups}

def format_html_document(document):
    """Expand generated markup so the checked-in HTML remains reviewable."""
    return document.replace('><','>\n<').rstrip()+'\n'

def build(root,output,web_dir=None):
    assets=root/'docs'; doc=assets/'zh'
    languages={language:build_language(root,language) for language in ('zh','en')}
    assert set(languages['zh']['pages'])==set(languages['en']['pages']), 'Language page sets differ'
    pages=languages['zh']['pages']; groups=languages['zh']['groups']
    payload=('\n'+json.dumps({'languages':languages,'icons':ICONS},ensure_ascii=False,indent=2).replace('<','\\u003c')+'\n')
    logo=(doc/'assets/aegaeon-mark.svg').read_text(encoding='utf-8').replace('xmlns="http://www.w3.org/2000/svg"','aria-hidden="true"')
    css=(assets/'portable.css').read_text(encoding='utf-8')+'\n'+HtmlFormatter(style='friendly').get_style_defs('.highlight')
    js=(assets/'portable.js').read_text(encoding='utf-8')
    tabs=''.join(f'<a href="#/zh/{group["first"]}" data-group="{group["id"]}">{html.escape(group["title"])}</a>' for group in groups)
    shell='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Aegaeon 模型即服务（MaaS）文档：以 Token 级自动扩缩容驱动高效服务，涵盖安装、部署、在线推理、架构与 API 参考。"><title>Aegaeon 文档</title>STYLE</head><body class="home"><header><div class="header-inner"><div class="header-top"><button class="mobile-toggle" id="mobile-toggle" aria-label="展开文档导航" aria-controls="sidebar" aria-expanded="false">MENU</button><a class="brand" href="#/index">LOGO<span>Aegaeon</span><span class="docs-label">文档</span></a><button class="search-trigger" id="search-trigger" aria-label="搜索文档">SEARCH<span>搜索文档</span><kbd>Ctrl K</kbd></button><a class="repo-link" href="#/architecture">SOURCE<span>系统架构<small>Aegaeon</small></span></a></div><nav class="nav-tabs" aria-label="主要导航">TABS</nav></div></header><div class="sidebar-shade" id="sidebar-shade"></div><div class="layout"><aside class="sidebar" id="sidebar" aria-label="文档导航"></aside><main class="main-column"><div class="breadcrumbs" id="breadcrumbs" aria-label="当前位置"></div><article id="article"></article><nav class="page-nav" id="page-nav" aria-label="文章导航"></nav><footer class="site-footer">Aegaeon · Apache-2.0</footer></main><aside class="toc" id="toc" aria-label="当前文章目录"></aside></div><dialog class="search-dialog" id="search-dialog" aria-label="搜索文档"><div class="search-field">SEARCH<input type="search" id="search-input" placeholder="搜索文档…" aria-label="搜索关键词" autocomplete="off"><button id="search-close" aria-label="关闭搜索">ESC</button></div><div class="search-results" id="search-results"></div><div class="search-footer">输入关键词搜索全文 <span aria-hidden="true"> · </span> Enter 打开结果 <span aria-hidden="true"> · </span> Esc 关闭</div></dialog><div class="toast" id="toast" role="status" aria-live="polite"></div><button class="back-top" id="back-top" title="返回顶部" aria-label="返回顶部">UP</button><script type="application/json" id="docs-data">PAYLOAD</script>SCRIPT</body></html>'''
    shell=shell.replace('<a class="repo-link"','<button class="language-toggle" id="language-toggle" title="Switch to English" aria-label="切换到 English">EN</button><a class="repo-link"',1)
    for old,new in [('MENU',ICONS['menu']),('LOGO',logo),('SEARCH',ICONS['search']),('GITHUB',GITHUB),('SOURCE',ICONS['link']),('TABS',tabs),('UP',ICONS['arrow-up']),('PAYLOAD',payload)]: shell=shell.replace(old,new)
    output.parent.mkdir(parents=True,exist_ok=True)
    portable_html=shell.replace('STYLE','<style>'+css+'</style>').replace('SCRIPT','<script>'+js+'</script>')
    # Keep the standalone document untouched because it embeds raw CSS and
    # JavaScript; only the external-asset web build is safe to expand by tags.
    output.write_text(portable_html,encoding='utf-8')
    if web_dir:
        web_dir.mkdir(parents=True,exist_ok=True)
        service_links=''.join(
            f'<a class="service-link" href="{asset}">{label}</a>'
            for asset,label in (('aegaeon_console.html','控制台'),('aegaeon_chat.html','对话'))
            if (web_dir/asset).is_file()
        )
        web_shell=shell.replace('<a class="repo-link"', service_links+'<a class="repo-link"',1)
        web_html=web_shell.replace('STYLE','<link rel="stylesheet" href="aegaeon_docs.css">').replace('SCRIPT','<script src="aegaeon_docs.js"></script>')
        (web_dir/'aegaeon_docs.html').write_text(format_html_document(web_html),encoding='utf-8')
        (web_dir/'aegaeon_docs.css').write_text(css,encoding='utf-8')
        (web_dir/'aegaeon_docs.js').write_text(js,encoding='utf-8')
    print(json.dumps({'pages_per_language':len(pages),'languages':list(languages),'groups':len(groups),'output':str(output),'bytes':output.stat().st_size},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--web-dir',type=Path)
    args=parser.parse_args(); build(args.root.resolve(),args.output.resolve(),args.web_dir)

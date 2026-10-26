(() => {
  const data = JSON.parse(document.getElementById('docs-data').textContent);
  const strings = {
    zh: {docs:'文档',home:'首页',toc:'目录',search:'搜索文档',keywords:'搜索关键词',close:'关闭搜索',menu:'展开文档导航',navigation:'文档导航',mainNavigation:'主要导航',location:'当前位置',articleNavigation:'文章导航',articleToc:'当前文章目录',copyCode:'复制代码',copyLink:'复制页面链接',print:'打印当前文章',prev:'上一篇',next:'下一篇',architecture:'系统架构',backTop:'返回顶部',copied:'已复制',copyFallback:'请选中内容后复制',searchEmpty:'搜索功能、参数、接口或错误信息。',notFound:'没有找到相关内容，请尝试其他关键词。',searchFooter:'输入关键词搜索全文 · Enter 打开结果 · Esc 关闭',console:'控制台',chat:'对话',switchLanguage:'切换到 English'},
    en: {docs:'Docs',home:'Home',toc:'On this page',search:'Search docs',keywords:'Search keywords',close:'Close search',menu:'Open documentation navigation',navigation:'Documentation navigation',mainNavigation:'Main navigation',location:'Current location',articleNavigation:'Article navigation',articleToc:'Article contents',copyCode:'Copy code',copyLink:'Copy page link',print:'Print this article',prev:'Previous',next:'Next',architecture:'Architecture',backTop:'Back to top',copied:'Copied',copyFallback:'Select the text and copy it manually',searchEmpty:'Search features, parameters, endpoints, or errors.',notFound:'No results. Try a different keyword.',searchFooter:'Search the full text · Enter to open · Esc to close',console:'Console',chat:'Chat',switchLanguage:'切换到中文'}
  };
  let language = 'zh';
  let preferredLanguage = 'zh';
  try { if (localStorage.getItem('aegaeon-docs-language') === 'en') preferredLanguage = 'en'; } catch {}
  let pages = data.languages.zh.pages;
  let groups = data.languages.zh.groups;
  const t = key => strings[language][key];
  const href = (id, anchor='') => `#/${language}/${id}${anchor ? '#' + encodeURIComponent(anchor) : ''}`;
  const article = document.getElementById('article');
  const sidebar = document.getElementById('sidebar');
  const toc = document.getElementById('toc');
  const breadcrumbs = document.getElementById('breadcrumbs');
  const pageNav = document.getElementById('page-nav');
  const dialog = document.getElementById('search-dialog');
  const searchInput = document.getElementById('search-input');
  const results = document.getElementById('search-results');
  const toggle = document.getElementById('mobile-toggle');
  let activeId = 'index';
  let observer;
  let toastTimer;
  let order = groups.flatMap(group => group.sections.flatMap(section => section.pages));
  const esc = value => String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  const icon = name => data.icons[name] || '';
  function toast(message) {
    const el = document.getElementById('toast'); el.textContent = message; el.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove('show'), 1800);
  }
  function closeSidebar() { document.body.classList.remove('sidebar-open'); toggle.setAttribute('aria-expanded', 'false'); }
  function renderSidebar(group) {
    sidebar.innerHTML = `<div class="sidebar-title">${esc(group.title)}</div>` + group.sections.map(section => {
      const links = section.pages.map(id => `<a href="${href(id)}"${id === activeId ? ' class="active" aria-current="page"' : ''}>${esc(pages[id].label)}</a>`).join('');
      if (!section.title) return `<nav aria-label="${esc(group.title)}">${links}</nav>`;
      return `<details open><summary>${esc(section.title)}${icon('chevron-down')}</summary><nav aria-label="${esc(section.title)}">${links}</nav></details>`;
    }).join('');
  }
  function renderToc(page) {
    toc.innerHTML = `<div class="toc-title">${t('toc')}</div>` + page.headings.map(h => `<a href="${href(activeId,h.id)}" class="depth-${h.level}">${esc(h.title)}</a>`).join('');
    if (observer) observer.disconnect();
    const links = [...toc.querySelectorAll('a')];
    observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top);
      if (visible.length) links.forEach(link => link.classList.toggle('active', decodeURIComponent(link.hash.split('#')[2] || '') === visible[0].target.id));
    }, {rootMargin:'-125px 0px -65% 0px', threshold:0});
    article.querySelectorAll('h2[id],h3[id]').forEach(heading => observer.observe(heading));
  }
  async function copy(text) {
    try {
      if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
      else {
        const field = document.createElement('textarea'); field.value = text;
        field.style.position = 'fixed'; field.style.opacity = '0'; document.body.appendChild(field); field.select();
        if (!document.execCommand('copy')) throw new Error('copy failed'); field.remove();
      }
      toast(t('copied'));
    } catch { toast(t('copyFallback')); }
  }
  function decorateArticle() {
    article.querySelectorAll('table').forEach(table => { const wrap = document.createElement('div'); wrap.className = 'table-wrap'; table.before(wrap); wrap.appendChild(table); });
    article.querySelectorAll('pre').forEach(pre => {
      const button = document.createElement('button'); button.className = 'code-copy'; button.title = t('copyCode'); button.setAttribute('aria-label',t('copyCode')); button.innerHTML = icon('copy');
      const code = pre.querySelector('code'); const text = (code || pre).textContent;
      button.addEventListener('click', () => copy(text));
      if (pre.parentElement.classList.contains('highlight')) pre.parentElement.appendChild(button); else pre.appendChild(button);
    });
    article.querySelectorAll('a[href^="http"]').forEach(link => { link.target = '_blank'; link.rel = 'noopener noreferrer'; });
  }
  function route() {
    let [path,anchor] = location.hash.replace(/^#\/?/, '').split('#');
    const parts = path.split('/');
    const requestedLanguage = data.languages[parts[0]] ? parts.shift() : preferredLanguage;
    const changedLanguage = language !== requestedLanguage;
    language = requestedLanguage; preferredLanguage = language;
    pages = data.languages[language].pages; groups = data.languages[language].groups;
    order = groups.flatMap(group => group.sections.flatMap(section => section.pages));
    let id = parts.join('/');
    id = ({home:'index', 'user-guide':'quickstart', 'developer-guide':'architecture', 'api-reference':'api', advanced:'architecture', 'live-demo':'serving'})[id] || id;
    if (!pages[id]) id = 'index';
    const samePage = activeId === id && article.dataset.loaded && !changedLanguage;
    activeId = id;
    const page = pages[id]; const group = groups.find(group => group.id === page.group);
    document.title = `${id === 'index' ? 'Aegaeon ' + t('docs') : page.label + ' - Aegaeon'}`;
    updateInterface();
    try { localStorage.setItem('aegaeon-docs-language',language); } catch {}
    if (!data.languages[path.split('/')[0]]) {
      try { history.replaceState(null,'',href(id,anchor ? decodeURIComponent(anchor) : '')); } catch {}
    }
    document.body.classList.toggle('home', id === 'index');
    document.querySelectorAll('.nav-tabs a').forEach(link => {
      const active = link.dataset.group === page.group;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current');
    });
    if (!samePage) {
      renderSidebar(group);
      breadcrumbs.innerHTML = `<a href="${href('index')}">${t('home')}</a>${icon('chevron-right')}<a href="${href(group.first)}">${esc(group.title)}</a>${icon('chevron-right')}<span>${esc(page.label)}</span><div class="article-tools"><button id="copy-link" title="${t('copyLink')}" aria-label="${t('copyLink')}">${icon('link')}</button><button id="print-page" title="${t('print')}" aria-label="${t('print')}">${icon('print')}</button></div>`;
      article.innerHTML = page.html; article.dataset.loaded = 'true'; decorateArticle(); renderToc(page);
      document.getElementById('copy-link').addEventListener('click', () => copy(location.href));
      document.getElementById('print-page').addEventListener('click', () => window.print());
      const position = order.indexOf(id); const prev = pages[order[position-1]]; const next = pages[order[position+1]];
      pageNav.innerHTML = (prev ? `<a href="${href(prev.id)}">${icon('arrow-left')}<span><small>${t('prev')}</small>${esc(prev.label)}</span></a>` : '') + (next ? `<a class="next" href="${href(next.id)}"><span><small>${t('next')}</small>${esc(next.label)}</span>${icon('arrow-right')}</a>` : '');
      if (!anchor) window.scrollTo({top:0,behavior:'instant'});
    }
    closeSidebar();
    if (anchor) {
      const target = document.getElementById(decodeURIComponent(anchor));
      if (target) requestAnimationFrame(() => target.scrollIntoView({behavior:'instant'}));
    }
  }
  function updateInterface() {
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelector('meta[name="description"]').content = language === 'zh' ? 'Aegaeon 多模型推理引擎文档：安装、部署、在线推理、架构与 API 参考。' : 'Aegaeon multi-model inference documentation: installation, deployment, online inference, architecture, and API references.';
    document.querySelector('.docs-label').textContent = t('docs');
    document.querySelector('.brand').href = href('index');
    document.querySelector('.search-trigger span').textContent = t('search');
    document.querySelector('.search-trigger').setAttribute('aria-label',t('search'));
    searchInput.placeholder = t('search') + '…'; searchInput.setAttribute('aria-label',t('keywords'));
    dialog.setAttribute('aria-label',t('search'));
    document.querySelector('#search-close').setAttribute('aria-label',t('close'));
    document.querySelector('.search-footer').textContent = t('searchFooter');
    document.querySelector('.repo-link span').innerHTML = esc(t('architecture')) + '<small>Aegaeon</small>';
    toggle.setAttribute('aria-label',t('menu'));
    [['#sidebar','navigation'],['.nav-tabs','mainNavigation'],['#breadcrumbs','location'],['#page-nav','articleNavigation'],['#toc','articleToc']].forEach(([selector,key]) => document.querySelector(selector).setAttribute('aria-label',t(key)));
    document.querySelector('#back-top').title = t('backTop');
    document.querySelector('#back-top').setAttribute('aria-label',t('backTop'));
    document.querySelectorAll('.service-link').forEach(link => { link.textContent = t(link.getAttribute('href').includes('console') ? 'console' : 'chat'); });
    const languageToggle = document.querySelector('#language-toggle');
    languageToggle.textContent = language === 'zh' ? 'EN' : '中文';
    languageToggle.title = t('switchLanguage'); languageToggle.setAttribute('aria-label',t('switchLanguage'));
    document.querySelector('.nav-tabs').innerHTML = groups.map(group => `<a href="${href(group.first)}" data-group="${group.id}">${esc(group.title)}</a>`).join('');
    if (dialog.open) search();
  }
  document.querySelector('#language-toggle').addEventListener('click', () => {
    const nextLanguage = language === 'zh' ? 'en' : 'zh';
    const currentAnchor = decodeURIComponent(location.hash.split('#')[2] || '');
    const index = pages[activeId].headings.findIndex(heading => heading.id === currentAnchor);
    const nextAnchor = index >= 0 ? data.languages[nextLanguage].pages[activeId].headings[index]?.id : '';
    location.hash = `#/${nextLanguage}/${activeId}${nextAnchor ? '#' + encodeURIComponent(nextAnchor) : ''}`;
  });
  function highlight(text,query) {
    const start = text.toLocaleLowerCase().indexOf(query.toLocaleLowerCase());
    if (start < 0) return esc(text);
    return esc(text.slice(0,start)) + '<mark>' + esc(text.slice(start,start+query.length)) + '</mark>' + esc(text.slice(start+query.length));
  }
  function search() {
    const query = searchInput.value.trim();
    if (!query) {
      results.innerHTML = `<div class="search-empty">${t('searchEmpty')}</div>`; return;
    }
    const terms = query.toLocaleLowerCase().split(/\s+/);
    const found = Object.values(pages).filter(page => terms.every(term => (page.label + ' ' + page.text).toLocaleLowerCase().includes(term)))
      .sort((a,b) => Number(b.label.toLocaleLowerCase().includes(query.toLocaleLowerCase())) - Number(a.label.toLocaleLowerCase().includes(query.toLocaleLowerCase()))).slice(0,12);
    results.innerHTML = found.length ? found.map(page => {
      const at = page.text.toLocaleLowerCase().indexOf(terms[0]); const begin = Math.max(0, at - 35);
      const snippet = (begin ? '…' : '') + page.text.slice(begin, begin+145) + '…';
      const group = groups.find(group => group.id === page.group);
      return `<a class="search-result" href="${href(page.id)}"><span class="result-category">${esc(group.title)}</span><strong>${highlight(page.label,query)}</strong><p>${highlight(snippet,query)}</p></a>`;
    }).join('') : `<div class="search-empty">${t('notFound')}</div>`;
  }
  function openSearch() { closeSidebar(); if (!dialog.open) dialog.showModal(); searchInput.focus(); searchInput.select(); search(); }
  document.getElementById('search-trigger').addEventListener('click', openSearch);
  document.getElementById('search-close').addEventListener('click', () => dialog.close());
  searchInput.addEventListener('input', search);
  results.addEventListener('click', event => { if (event.target.closest('a')) dialog.close(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) { const r = dialog.getBoundingClientRect(); if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close(); }});
  searchInput.addEventListener('keydown',event => {
    if (event.key === 'Enter') { const first = results.querySelector('a'); if (first) { location.hash = first.hash; dialog.close(); } }
    if (event.key === 'ArrowDown') { const first = results.querySelector('a'); if (first) { event.preventDefault(); first.focus(); } }
  });
  document.addEventListener('keydown',event => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); openSearch(); }
    if (event.key === '/' && !/INPUT|TEXTAREA/.test(event.target.tagName) && !dialog.open) { event.preventDefault(); openSearch(); }
    if (event.key === 'Escape') closeSidebar();
  });
  toggle.addEventListener('click', () => { const open = document.body.classList.toggle('sidebar-open'); toggle.setAttribute('aria-expanded', String(open)); });
  document.getElementById('sidebar-shade').addEventListener('click',closeSidebar);
  const backTop = document.getElementById('back-top');
  backTop.addEventListener('click', () => window.scrollTo({top:0,behavior:'smooth'}));
  window.addEventListener('scroll', () => backTop.classList.toggle('visible', window.scrollY > 650), {passive:true});
  window.addEventListener('hashchange', route);
  route();
})();

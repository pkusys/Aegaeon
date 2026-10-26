"""Resolve documentation links inside the homepage's HTML components."""
import html as html_module
import posixpath
import re

from mkdocs.utils import get_relative_url


def on_page_content(html, page, config, files):
    def resolve(match):
        target = html_module.unescape(match.group(1))
        path, separator, anchor = target.partition('#')
        if not path.endswith('.md') or '://' in path:
            return match.group(0)
        source = posixpath.normpath(posixpath.join(posixpath.dirname(page.file.src_uri), path))
        destination = files.get_file_from_path(source)
        if destination is None:
            return match.group(0)
        url = get_relative_url(destination.url, page.file.url)
        if separator:
            url += '#' + anchor
        return 'href="' + html_module.escape(url, quote=True) + '"'

    return re.sub(r'href="([^"]+)"', resolve, html)

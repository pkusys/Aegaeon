from pathlib import Path
import ast
import json
import os
import re
import shlex
import subprocess
import sys
import yaml

root=Path(sys.argv[1] if len(sys.argv)>1 else str(Path(__file__).resolve().parents[1]))
docs=[root/'docs'/language for language in ('zh','en')]
errors=[]
counts={'markdown':0,'links':0,'yaml_files':0,'yaml_blocks':0,'json_blocks':0,'python_blocks':0,'cli_examples':0,'scope_checks':0,'shell_blocks':0,'shell_inputs':0,'json_files':0}
# These are file roles, not shell evaluation. In particular, model IDs and
# output arguments are not evidence that an input asset belongs in the checkout.
_SHELL_INPUT_OPTIONS = {'--config', '--config-file', '--model-config',
                        '--input', '--input-file', '--request-file', '--workload-file'}
_SHELL_OUTPUT_OPTIONS = {'--output', '--output-file', '--output-json', '--output-dir',
                         '--result-file', '--web-dir', '--site-dir', '--local-dir'}
_CURL_DATA_OPTIONS = {'-d', '--data', '--data-ascii', '--data-binary', '--data-raw',
                      '--data-urlencode', '--json'}


def shell_asset_path(reference):
    """Return a concrete checkout path; leave external/parameterized paths alone."""
    if not reference or reference == '-' or re.search(r'[$`~*?\[\]{}<>]', reference):
        return None
    if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', reference):
        return None
    if re.fullmatch(r'[A-Z][A-Z0-9_]*', reference):
        return None
    path = Path(reference.split('::', 1)[0])
    if path.is_absolute():
        return None
    path = (root / path).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        return None
    return path


def check_shell_command(tokens, document, line, generated):
    while tokens and (tokens[0] == 'env' or re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', tokens[0])):
        tokens = tokens[1:]
    if not tokens:
        return
    command = tokens[0] if tokens[0] == '.' else Path(tokens[0]).name
    python = bool(re.fullmatch(r'python(?:\d+(?:\.\d+)*)?', command))
    module = None
    positional, inputs, outputs = [], [], []
    args = tokens[1:]
    index = 0
    while index < len(args):
        token = args[index]
        option, equal, value = token.partition('=')
        if command == 'curl' and token.startswith('-d') and not token.startswith('--') and token != '-d':
            option, equal, value = '-d', True, token[2:]
        takes_input = option in _SHELL_INPUT_OPTIONS
        takes_input |= option in {'-r', '--requirement', '-c', '--constraint', '-e', '--editable'} and (command == 'pip' or module == 'pip')
        takes_input |= option == '-f' and (command == 'mkdocs' or module == 'mkdocs')
        takes_input |= option in {'-f', '-e', '-d'} and command in {'test', '['}
        takes_output = option in _SHELL_OUTPUT_OPTIONS or (command == 'curl' and option in {'-o', '--output'})
        curl_data = command == 'curl' and option in _CURL_DATA_OPTIONS
        ignores_value = (python and option in {'-c', '-m', '-W', '-X'}) or (command in {'bash', 'sh', 'zsh'} and option == '-c')
        ignores_value |= module == 'pytest' and option in {'-k', '-m', '--maxfail'}
        ignores_value |= command in {'head', 'tail'} and option in {'-n', '--lines', '-c', '--bytes'}
        if token.isdigit() and index + 1 < len(args) and args[index + 1] in {'<', '>', '>>', '<&', '>&'}:
            index += 1  # A shell file descriptor, not a positional filename.
            continue
        if token in {'<&', '>&', '<<<'}:
            index += 2  # Descriptor duplication / a literal here-string.
            continue
        if token in {'<', '>', '>>', '&>'}:
            if index + 1 < len(args):
                (inputs if token == '<' else outputs).append(args[index + 1])
            index += 2
            continue
        if takes_input or takes_output or curl_data or ignores_value:
            if not equal:
                index += 1
                value = args[index] if index < len(args) else ''
            if python and option == '-m' and module is None:
                module = value
            elif takes_input:
                inputs.append(value)
            elif takes_output:
                outputs.append(value)
            elif curl_data and option != '--data-raw':
                # --data-raw deliberately treats @ as literal request text.
                if value.startswith('@'):
                    inputs.append(value[1:])
                elif option == '--data-urlencode' and '@' in value:
                    inputs.append(value.partition('@')[2])
            index += 1
            continue
        if not token.startswith('-'):
            positional.append(token)
        index += 1

    if python and module is None and '-c' not in args and positional:
        inputs.append(positional[0])
    elif python and module == 'pytest':
        inputs.extend(positional)
    elif command in {'bash', 'sh', 'zsh', 'source', '.'} and '-c' not in args and positional:
        inputs.append(positional[0])
    elif command in {'cat', 'head', 'tail', 'less'}:
        inputs.extend(positional)
    elif command == 'cp' and len(positional) >= 2:
        inputs.extend(positional[:-1])
        outputs.append(positional[-1])
    elif command == 'sed' and '-i' in args and len(positional) >= 2:
        inputs.extend(positional[1:])
    elif '/' in tokens[0]:
        inputs.append(tokens[0])

    for reference in inputs:
        asset = shell_asset_path(reference)
        if asset is None or asset in generated:
            continue
        counts['shell_inputs'] += 1
        if not asset.exists():
            label = document.relative_to(root).as_posix()
            errors.append(f'{label}:{line}: missing shell input {reference!r}')
    for reference in outputs:
        asset = shell_asset_path(reference)
        if asset is not None:
            generated.add(asset)


def check_shell_block(block, document, first_line, generated):
    """Tokenize commands without running them, keeping physical source locations."""
    counts['shell_blocks'] += 1
    pending = ''
    command_line = first_line
    for offset, physical_line in enumerate(block.splitlines() + [''], first_line):
        if not pending:
            command_line = offset
        if physical_line.endswith('\\'):
            pending += physical_line[:-1] + ' '
            continue
        pending += physical_line
        lexer = shlex.shlex(pending, posix=True, punctuation_chars=';&|<>')
        lexer.whitespace_split = True
        try:
            tokens = list(lexer)
        except ValueError as exc:
            errors.append(f'{document.relative_to(root).as_posix()}:{command_line}: shell tokenization failed: {exc}')
            pending = ''
            continue
        command = []
        for token in tokens + [';']:
            if token and set(token) <= set(';&|'):
                check_shell_command(command, document, command_line, generated)
                command = []
            else:
                command.append(token)
        pending = ''


def fields(source,cls):
    node=next(x for x in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(x,ast.ClassDef) and x.name==cls)
    return {x.target.id for x in node.body if isinstance(x,ast.AnnAssign)}

def constant(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(constant(item) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -constant(node.operand)
    if isinstance(node, ast.BinOp):
        left, right = constant(node.left), constant(node.right)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.Pow: lambda: left**right,
        }
        return operations[type(node.op)]()
    raise ValueError(ast.dump(node))

def defaults(source, cls):
    node=next(x for x in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(x,ast.ClassDef) and x.name==cls)
    result={}
    for item in node.body:
        if isinstance(item,ast.AnnAssign) and item.value is not None:
            try: result[item.target.id]=constant(item.value)
            except (KeyError,TypeError,ValueError): pass
    return result

def mask_fenced_code(text):
    """Hide fenced examples while preserving their physical source locations."""
    return re.sub(
        r'```[^\n]*\n.*?```',
        lambda match: re.sub(r'[^\n]', ' ', match.group(0)),
        text,
        flags=re.S,
    )


def markdown_table(text, start_heading):
    prose = mask_fenced_code(text)
    heading = re.fullmatch(r'(#{1,6})\s+(.+)', start_heading)
    headings = list(re.finditer(r'^(#{1,6})[ \t]+(.+?)[ \t]*$', prose, flags=re.M))
    start = next(
        (index for index, match in enumerate(headings)
         if match.group(1) == heading.group(1) and match.group(2) == heading.group(2)),
        None,
    )
    if start is None:
        return {}
    level = len(headings[start].group(1))
    end = next(
        (match.start() for match in headings[start + 1:] if len(match.group(1)) <= level),
        len(prose),
    )
    section = prose[headings[start].end():end]
    result = {}
    for line in section.splitlines():
        match = re.match(r'\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|', line)
        if match and not set(match.group(1).strip()) <= {'-', ':'}:
            result[match.group(1).strip()] = match.group(2).strip()
    return result


def shown(value):
    if value is None: return 'null'
    if value is True: return 'true'
    if value is False: return 'false'
    if isinstance(value,tuple): return '['+','.join(str(item) for item in value)+']'
    if isinstance(value,float): return str(round(value,12)).rstrip('0').rstrip('.')
    return str(value)
server_fields=fields(root/'aegaeon/config.py','ServerConfig')
model_fields={'name','path','params','profile','max_model_len','id'}
graph_fields=fields(root/'aegaeon/graph_config.py','CudaGraphConfig')
server_defaults=defaults(root/'aegaeon/config.py','ServerConfig')
graph_defaults=defaults(root/'aegaeon/graph_config.py','CudaGraphConfig')

def check_yaml(data,where):
    if not isinstance(data,dict): return
    if 'server' in data:
        server=data['server']
        unknown=set(server)-server_fields
        if unknown: errors.append(f'{where}: unknown server fields {unknown}')
        graph=server.get('cuda_graph')
        if graph:
            if set(graph)-graph_fields: errors.append(f'{where}: unknown graph fields')
            if not isinstance(graph.get('mode','off'),str): errors.append(f'{where}: graph mode not string')
            if graph.get('mode') in ('save','load') and len(graph.get('seq_len_buckets',[]))!=1: errors.append(f'{where}: save/load requires one bucket')
        if server.get('num_engines',0)>0 and (server.get('num_prefill_engines',0)!=0 or server.get('num_decode_engines',0)!=0): errors.append(f'{where}: modes conflict')
        if server.get('tensor_parallel_size',1) != 1:
            errors.append(f'{where}: only tensor_parallel_size=1 is supported')
    for item in data.get('models',[]):
        if set(item)-model_fields: errors.append(f'{where}: unknown model fields')

page_sets=[{path.relative_to(doc).as_posix() for path in doc.rglob('*.md')} for doc in docs]
if page_sets[0]!=page_sets[1]: errors.append('Chinese and English page sets differ')


def navigation_pages(items):
    for item in items:
        for value in item.values():
            if isinstance(value, list):
                yield from navigation_pages(value)
            else:
                yield value


for language, page_set in zip(('zh', 'en'), page_sets):
    config = yaml.load((root / f'mkdocs.{language}.yml').read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    navigation = list(navigation_pages(config['nav']))
    if len(navigation) != len(set(navigation)):
        errors.append(f'{language} navigation contains duplicate pages')
    if set(navigation) != page_set:
        errors.append(
            f'{language} navigation: missing files {sorted(set(navigation)-page_set)}, '
            f'unlisted pages {sorted(page_set-set(navigation))}'
        )
for path in sorted(path for doc in docs for path in doc.rglob('*.md')):
    counts['markdown']+=1
    text=path.read_text(encoding='utf-8')
    if text.count('```')%2: errors.append(f'{path.name}: unbalanced fences')
    # Mask code, rather than deleting it, so link diagnostics retain line numbers.
    prose = mask_fenced_code(text)
    for match in re.finditer(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', prose):
        target = match.group(1)
        if target.startswith(('http:', 'https:', '#', 'mailto:')):
            continue
        counts['links'] += 1
        resolved = (path.parent / target.split('#')[0]).resolve()
        if not resolved.exists():
            line = prose[:match.start()].count('\n') + 1
            label = path.relative_to(root).as_posix()
            resolved_label = Path(os.path.relpath(resolved, root)).as_posix()
            errors.append(f'{label}:{line}: missing link {target!r} (resolved {resolved_label})')
    generated = set()
    for fence in re.finditer(r'```([^\n]*)\n(.*?)```', text, flags=re.S):
        lang, block = fence.groups()
        if lang.strip() in {'bash', 'sh', 'shell', 'console'}:
            first_line = text[:fence.start(2)].count('\n') + 1
            check_shell_block(block, path, first_line, generated)
        try:
            if lang.strip()=='python': ast.parse(block); counts['python_blocks']+=1
            elif lang.strip()=='json': json.loads(block); counts['json_blocks']+=1
            elif lang.strip() in ('yaml','yml'): check_yaml(yaml.safe_load(block),path.name); counts['yaml_blocks']+=1
        except Exception as exc: errors.append(f'{path.name}: {lang} invalid: {exc}')
for path in (path for doc in docs for path in (doc/'examples').glob('*.yaml')):
    counts['yaml_files']+=1
    try: check_yaml(yaml.safe_load(path.read_text(encoding='utf-8')),path.name)
    except Exception as exc: errors.append(f'{path.name}: {exc}')
for path in (path for doc in docs for path in (doc/'examples').glob('*.json')):
    counts['json_files']+=1
    try: json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc: errors.append(f'{path.relative_to(root)}: JSON invalid: {exc}')

# Product boundaries are deliberate even where parsers or reserved code paths
# accept broader shapes. Keep every user-facing entry point on the same scope.
zh_all='\n'.join(path.read_text(encoding='utf-8') for path in docs[0].rglob('*.md'))
en_all='\n'.join(path.read_text(encoding='utf-8') for path in docs[1].rglob('*.md'))
en_coming_next=(docs[1]/'capabilities.md').read_text(encoding='utf-8').partition('**Coming Next**:')[2].split('\n## ',1)[0]
scope_requirements = (
    ('zh TP scope', '仅支持 `tensor_parallel_size=1`' in zh_all),
    ('en TP scope', 'Only `tensor_parallel_size=1` is supported' in en_all or
     'Currently supported configuration assumes TP=1' in (docs[1]/'deployment.md').read_text(encoding='utf-8')),
    ('zh multimodal scope', '多模态输入 | 当前仅提供文本输入与文本生成' in zh_all),
    ('en multimodal scope', 'Multimodal input | **Unsupported**' in en_all or
     re.search(r'^- Multimodal input\s*$', en_coming_next, flags=re.M) is not None),
    ('estimator TP assertion', 'assert tp == 1' in (root/'aegaeon/estimator.py').read_text(encoding='utf-8')),
    ('pipeline assertion', 'assert pipeline_parallel_size == 1' in (root/'aegaeon/config.py').read_text(encoding='utf-8')),
    ('greedy implementation', 'torch.argmax(logits, dim=-1)' in (root/'aegaeon/worker.py').read_text(encoding='utf-8')),
)
for name, passed in scope_requirements:
    counts['scope_checks'] += 1
    if not passed:
        errors.append(f'scope check failed: {name}')

for stale in (
    r'TP>1.{0,40}(验证|测试|check|test|validat)',
    r'(多模态支持|multimodal support).{0,40}(取决于|depend)',
):
    if re.search(stale, zh_all+'\n'+en_all, flags=re.I):
        errors.append(f'stale support wording: {stale}')

def cell_value(cell, template):
    raw=cell.replace('`','').strip()
    if isinstance(template,bool): return raw.lower()=='true'
    if template is None: return None if raw.lower()=='null' else raw
    if isinstance(template,tuple):
        return tuple(int(item.strip()) for item in raw.strip('[]').split(',') if item.strip())
    if isinstance(template,int): return int(raw,0)
    if isinstance(template,float): return float(raw)
    if isinstance(template,str): return raw.strip("'\"")
    return raw

reference_tables = (
    (docs[0]/'configuration.md','## ServerConfig 默认值',server_defaults,'zh ServerConfig'),
    (docs[1]/'configuration.md','## ServerConfig defaults',server_defaults,'en ServerConfig'),
    (docs[0]/'configuration.md','## cuda_graph 默认与约束',graph_defaults,'zh CudaGraphConfig'),
    (docs[1]/'configuration.md','## CUDA graph defaults and constraints',graph_defaults,'en CudaGraphConfig'),
)
for path,start,source_defaults,label in reference_tables:
    table=markdown_table(path.read_text(encoding='utf-8'),start)
    for field,value in source_defaults.items():
        counts['scope_checks'] += 1
        if field not in table:
            errors.append(f'{label}: missing default row {field}')
            continue
        try:
            documented=cell_value(table[field],value)
        except Exception as exc:
            errors.append(f'{label}: cannot parse {field}: {exc}')
            continue
        if documented != value:
            errors.append(f'{label}: {field} default {documented!r} != source {value!r}')

api_source=root/'aegaeon/api.py'
public_routes=set()
for node in ast.parse(api_source.read_text(encoding='utf-8')).body:
    if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)): continue
    for decorator in node.decorator_list:
        if not isinstance(decorator,ast.Call) or not decorator.args: continue
        func=decorator.func
        if not (
            isinstance(func,ast.Attribute) and
            isinstance(func.value,ast.Name) and func.value.id=='app' and
            func.attr in {'get','post','put','patch','delete'} and
            isinstance(decorator.args[0],ast.Constant)
        ): continue
        hidden=any(
            keyword.arg=='include_in_schema' and isinstance(keyword.value,ast.Constant) and keyword.value.value is False
            for keyword in decorator.keywords
        )
        if not hidden:
            public_routes.add((func.attr.upper(),decorator.args[0].value))
for path,label in (
    (docs[0]/'api.md','zh routes'),
    (docs[1]/'api.md','en routes'),
):
    prose = mask_fenced_code(path.read_text(encoding='utf-8'))
    documented_routes=set(re.findall(r'^\|\s*(GET|POST|PUT|PATCH|DELETE)\s*\|\s*(`?[^|`]+`?)\s*\|',prose,flags=re.M))
    documented_routes={(method,route.strip().strip('`')) for method,route in documented_routes}
    counts['scope_checks'] += 1
    if documented_routes != public_routes:
        errors.append(
            f'{label}: missing {sorted(public_routes-documented_routes)}, '
            f'extra {sorted(documented_routes-public_routes)}'
        )

# Help commands are independent of GPU imports because they terminate argparse first.
for cmd in ([],['doctor'],['start'],['deploy'],['undeploy']):
    result=subprocess.run([sys.executable,str(root/'aegaeon/cli.py'),*cmd,'--help'],capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode: errors.append(f'CLI help {cmd} failed: {result.stderr}')
    counts['cli_examples']+=1

for command in ('doctor', 'start', 'deploy', 'undeploy'):
    result=subprocess.run(
        [sys.executable,str(root/'aegaeon/cli.py'),command,'--help'],
        capture_output=True,text=True,encoding='utf-8',errors='replace'
    )
    source_options=set(re.findall(r'(?<!\w)--[a-z][a-z-]+',result.stdout))-{'--help'}
    for path,label in (
        (docs[0]/'cli.md','zh CLI'),
        (docs[1]/'cli.md','en CLI'),
    ):
        table=markdown_table(path.read_text(encoding='utf-8'), f'## {command}')
        documented={name for name in table if name.startswith('--')}
        counts['scope_checks'] += 1
        if documented != source_options:
            errors.append(
                f'{label} {command}: documented options {sorted(documented)} '
                f'!= parser {sorted(source_options)}'
            )

print(json.dumps({'passed':not errors,'counts':counts,'errors':errors},ensure_ascii=False,indent=2))
sys.exit(bool(errors))

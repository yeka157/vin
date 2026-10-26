"""Core behavior checks for workspace file tools and Bash."""

import importlib
import json
import os
from pathlib import Path
import shlex
import stat
import sys
import tempfile
import time
from threading import Event, Thread
import unittest
from unittest.mock import patch


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        previous = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)
        # A fresh import models a separate launch without leaking state to tests.
        previous_module = sys.modules.pop('vin.tools', None)
        package = importlib.import_module('vin')
        previous_attribute = getattr(package, 'tools', None)
        def restore_module():
            sys.modules.pop('vin.tools', None)
            if previous_module is not None:
                sys.modules['vin.tools'] = previous_module
            if previous_attribute is not None:
                package.tools = previous_attribute
            elif hasattr(package, 'tools'):
                del package.tools
        self.addCleanup(restore_module)
        self.assertIsNotNone(importlib.util.find_spec('vin.tools'),
                             'Implement the four public functions in src/vin/tools.py first.')
        self.tools = importlib.import_module('vin.tools')

    def rejected(self, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
        except (ValueError, OSError, RuntimeError, UnicodeError):
            return
        self.assertRegex(str(result).lower(), r'error|reject|ambiguous|not found|no change|limit|exceed',
                         'Invalid operations must raise or return an explicit error.')

    def output(self, result):
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

    def assert_exit(self, result, code):
        self.assertRegex(self.output(result).lower(), rf'exit[_ ]code[\s"\x27:=]+{code}(?!\d)')


    def test_read_line_numbers_offset_limit_and_default_200(self):
        (self.root/'f').write_text(''.join(f'line {n}\n' for n in range(1, 205)))
        result = self.output(self.tools.read('f', offset=2, limit=2))
        self.assertRegex(result, r'(?m)^\s*2\s*[:|\t ]\s*line 2$')
        self.assertRegex(result, r'(?m)^\s*3\s*[:|\t ]\s*line 3$')
        self.assertNotIn('line 4', result)
        default = self.output(self.tools.read('f'))
        self.assertIn('line 200', default)
        self.assertNotIn('line 201', default)
        for kwargs in [{'offset': 0}, {'offset': -1}, {'limit': 0}, {'limit': 2001}]:
            self.rejected(self.tools.read, 'f', **kwargs)
        (self.root/'f').write_text('x\n'*2001)
        self.assertEqual(len(self.output(self.tools.read('f', limit=2000)).splitlines()), 2000)

    def test_write_creates_parents_and_overwrites_utf8(self):
        self.tools.write('nested/f', 'café\n')
        file = self.root/'nested/f'
        self.assertEqual(file.read_bytes(), 'café\n'.encode())
        self.tools.write('nested/f', 'new')
        self.assertEqual(file.read_bytes(), b'new')

    def test_edit_returns_diff_and_exact_match_preserves_other_bytes(self):
        file = self.root/'f'
        file.write_bytes('keep “this”  \n  old  \nend\u00a0 \n'.encode())
        result = self.output(self.tools.edit('f', 'old', 'new'))
        self.assertIn('---', result)
        self.assertIn('+++', result)
        self.assertIn('@@', result)
        self.assertEqual(file.read_bytes(), 'keep “this”  \n  new  \nend\u00a0 \n'.encode())

    def test_fuzzy_preserves_bom_crlf_unrelated_lines_and_permissions(self):
        file = self.root/'f'
        file.write_bytes('\ufeffkeep “this”  \r\n  x = “café” – ２  \r\nuntouched\u00a0 \r\n'.encode())
        file.chmod(0o751)
        self.tools.edit('f', '  x = "cafe\u0301" - 2', '  x = "done"')
        self.assertEqual(file.read_bytes(), '\ufeffkeep “this”  \r\n  x = "done"\r\nuntouched\u00a0 \r\n'.encode())
        self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o751)
        self.tools.edit('f', 'done', 'finished\nnext')
        self.assertIn(b'finished\r\nnext', file.read_bytes())
        self.assertNotIn(b'\r\r\n', file.read_bytes())
        self.tools.write('f', 'new')
        self.assertEqual(stat.S_IMODE(file.stat().st_mode), 0o751)

    def test_fuzzy_matching_normalizes_punctuation_spaces_and_compatibility(self):
        for source, search in [('“hello”', '"hello"'), ('‘hello’', "'hello'"),
                               ('a—b', 'a-b'), ('a\u00a0b', 'a b'), ('a\u2009b', 'a b'),
                               ('ﬃ', 'ffi'), ('Ａ', 'A'), ('cafe\u0301', 'café'),
                               ('a  \nb\t\n', 'a\nb\n')]:
            with self.subTest(source=source):
                (self.root/'f').write_text(source, encoding='utf-8')
                self.tools.edit('f', search, 'changed')
                self.assertEqual((self.root/'f').read_text(), 'changed')

    def test_fuzzy_multiline_keeps_indentation_and_duplicate_unrelated_lines(self):
        file = self.root/'f'
        file.write_bytes('same “line”  \n  a = “old”  \n    b = １\nsame "line"\n'.encode())
        self.tools.edit('f', '  a = "old"\n    b = 1', '  a = "new"\n    b = 2')
        self.assertEqual(file.read_bytes(), 'same “line”  \n  a = "new"\n    b = 2\nsame "line"\n'.encode())
        original = file.read_bytes()
        self.rejected(self.tools.edit, 'f', '  a = “new”\n  b = 2', 'bad')
        self.assertEqual(file.read_bytes(), original)

    def test_edit_rejections_leave_file_unchanged(self):
        cases = [('abc', '', 'x'), ('abc', ' \n\t', 'x'), ('abc', 'missing', 'x'),
                 ('abc', 'abc', 'abc'), ('x x', 'x', 'y'),
                 ('café cafe\u0301', 'café', 'y'), ('“x” "x"', '"x"', 'y'),
                 ('Ａ A', 'A', 'y'), ('aaa', 'aa', 'b')]
        for content, old, new in cases:
            with self.subTest(content=content, old=old):
                file = self.root/'f'
                file.write_text(content, encoding='utf-8')
                self.rejected(self.tools.edit, 'f', old, new)
                self.assertEqual(file.read_bytes(), content.encode())

    def test_paths_and_symlinks_cannot_escape_launch_directory(self):
        with tempfile.TemporaryDirectory() as outside:
            external = Path(outside)/'f'
            external.write_text('private')
            (self.root/'link').symlink_to(outside, target_is_directory=True)
            (self.root/'file-link').symlink_to(external)
            relative_escape = os.path.relpath(external, self.root)
            for path in [str(external), relative_escape, 'link/f', 'link/new', 'file-link']:
                for method, args in [(self.tools.read, ()), (self.tools.write, ('x',)),
                                     (self.tools.edit, ('private', 'x'))]:
                    with self.subTest(path=path, method=method.__name__):
                        self.rejected(method, path, *args)
            self.assertEqual(external.read_text(), 'private')
            self.assertFalse((Path(outside)/'new').exists())

    def test_relative_paths_keep_original_launch_directory_after_chdir(self):
        (self.root/'f').write_text('launch')
        (self.root/'sub').mkdir()
        os.chdir(self.root/'sub')
        self.assertIn('launch', self.output(self.tools.read('f')))
        self.tools.write('new', 'launch write')
        self.assertTrue((self.root/'new').exists())
        self.assertFalse((self.root/'sub/new').exists())
        self.assertIn(str(self.root), self.output(self.tools.bash('pwd')))

    def test_two_mb_byte_limits_and_edit_growth(self):
        file = self.root/'f'
        file.write_bytes(b'x'*2_000_001)
        self.rejected(self.tools.read, 'f')
        self.rejected(self.tools.edit, 'f', 'x', 'y')
        self.rejected(self.tools.write, 'new', 'é'*1_000_001)
        self.assertFalse((self.root/'new').exists())
        file.write_bytes(b'a' + b'x'*1_999_999)
        self.rejected(self.tools.edit, 'f', 'a', 'é')
        self.assertEqual(file.stat().st_size, 2_000_000)
        self.tools.write('boundary', 'é'*1_000_000)
        self.assertEqual((self.root/'boundary').stat().st_size, 2_000_000)

    def test_tool_output_caps_have_visible_marker(self):
        (self.root/'f').write_text('x'*30_000)
        python = shlex.quote(sys.executable)
        for result in [self.tools.read('f'), self.tools.bash(f'{python} -c \'print("x"*40000)\'')]:
            output = self.output(result)
            self.assertLessEqual(len(output), 20_000)
            self.assertIn('truncat', output.lower())
        (self.root/'f').write_text('old')
        diff = self.output(self.tools.edit('f', 'old', 'x'*30_000))
        self.assertLessEqual(len(diff), 20_000)
        self.assertIn('truncat', diff.lower())
        self.assertEqual((self.root/'f').read_text(), 'x'*30_000)

    def test_bash_combines_stderr_and_stdout_exit_status_and_launch_cwd(self):
        result = self.tools.bash('printf out; printf err >&2; exit 7')
        self.assertIn('outerr', self.output(result))
        self.assert_exit(result, 7)
        self.assertIn(str(self.root), self.output(self.tools.bash('pwd')))
        self.assert_exit(self.tools.bash('read value || true'), 0)
        for timeout in [0, -1, 301]:
            self.rejected(self.tools.bash, 'true', timeout=timeout)

    def test_bash_timeout_terminates_descendants(self):
        marker = self.root/'late'
        start = time.monotonic()
        # An un-killed child would create late after its shell has been terminated.
        result = self.tools.bash('(sleep 1.5; printf leaked > late) & wait', timeout=1)
        self.assertLess(time.monotonic()-start, 4)
        self.assertIn('tim', self.output(result).lower())
        time.sleep(.7)
        self.assertFalse(marker.exists(), 'Timeout must kill the entire process group.')
        self.assertRegex(self.output(result).lower(), r'exit[_ ]code')

    def test_bash_is_not_limited_by_file_tool_path_checks(self):
        with tempfile.TemporaryDirectory() as outside:
            file = Path(outside)/'f'
            file.write_text('outside-data')
            self.assertIn('outside-data', self.output(self.tools.bash(f'cat {shlex.quote(str(file))}')))

    def test_atomic_replace_failure_keeps_original_and_cleans_temporary_file(self):
        file = self.root/'f'
        file.write_text('old')
        with patch('os.replace', side_effect=OSError('simulated rename failure')):
            self.rejected(self.tools.edit, 'f', 'old', 'new')
        self.assertEqual(file.read_text(), 'old')
        self.assertEqual({p.name for p in self.root.iterdir()}, {'f'})

    def test_aisuite_validates_arguments_and_keeps_python_defaults(self):
        from aisuite.utils.tools import Tools
        registry = Tools([self.tools.read, self.tools.write, self.tools.edit, self.tools.bash])
        (self.root/'f').write_text('one\ntwo')
        result, = registry.execute({'function': {'name': 'read', 'arguments': '{"path":"f"}'}})
        self.assertIn('one', self.output(result))
        self.assertIn('two', self.output(result))
        with self.assertRaises((ValueError, TypeError)):
            registry.execute({'function': {'name': 'read', 'arguments': '{"path":[]}'}})

    def test_concurrent_edits_are_serialized_without_lost_updates(self):
        file = self.root/'f'
        file.write_text('left right')
        committing, release, second_done = Event(), Event(), Event()
        results = []
        actual_replace = os.replace
        def delayed_replace(*args, **kwargs):
            if not committing.is_set():
                committing.set()
                if not release.wait(3):
                    raise RuntimeError('Test timed out waiting for release.')
            return actual_replace(*args, **kwargs)
        def edit(old, new, done=None):
            try:
                results.append(self.tools.edit('f', old, new))
            except Exception as error:
                results.append(error)
            finally:
                if done:
                    done.set()
        first = Thread(target=edit, args=('left', 'LEFT'), daemon=True)
        second = Thread(target=edit, args=('right', 'RIGHT', second_done), daemon=True)
        with patch('os.replace', side_effect=delayed_replace):
            try:
                first.start()
                self.assertTrue(committing.wait(2), 'Edit must commit through an atomic replacement.')
                second.start()
                self.assertFalse(second_done.wait(.1), 'A second mutation must wait for the first.')
            finally:
                release.set()
                if first.ident is not None:
                    first.join(3)
                if second.ident is not None:
                    second.join(3)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(any(isinstance(result, Exception) for result in results), results)
        self.assertEqual(file.read_text(), 'LEFT RIGHT')

    def test_external_change_before_save_is_rejected(self):
        file = self.root/'f'
        file.write_text('old')
        original = tempfile.mkstemp
        def changed(*args, **kwargs):
            file.write_text('external')
            return original(*args, **kwargs)
        # Injection point: standard-library temporary-file creation. If the
        # implementation uses a different atomic-write helper, adapt this hook.
        with patch('tempfile.mkstemp', side_effect=changed) as create:
            self.rejected(self.tools.edit, 'f', 'old', 'new')
        self.assertTrue(create.called, 'Adapt the injection hook if another temp-file API is used.')
        self.assertEqual(file.read_text(), 'external')


if __name__ == '__main__':
    unittest.main()

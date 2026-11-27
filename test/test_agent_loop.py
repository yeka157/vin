"""Offline Responses contract tests for the existing stream worker.

Real SDK parsing is exercised with HTTP mock streams; no account, credentials,
network, or implementation code is added. Tool execution is mocked through
aisuite.Tools.execute, independently of the user's local tool implementation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from threading import Event
from typing import TYPE_CHECKING, Literal, cast
import unittest
from unittest.mock import Mock, patch

from aisuite.utils.tools import Tools
import httpx2
import openai
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFunctionToolCallParam,
    ResponseOutputMessageParam,
    ResponseOutputRefusalParam,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document

from provider_typing_helpers import (
    REQUEST_ADAPTER,
    ClientDouble,
    HTTPFixture,
    OutputItem,
    RequestSnapshot,
    WireArgumentsDelta,
    WireCompleted,
    WireDelta,
    WireEvent,
    WireItemDone,
    credentials as fixture_credentials,
)
from vin import agent
from vin.provider import prompts
from vin.provider.chatgpt import stream_chatgpt_response
from vin.provider.events import ReplyEvents
from vin.types.events import ReplyEvent
import vin.tui as tui
from vin.types.auth import Credentials
from vin.types.data import JSONValue, is_string_list, json_value
from vin.types.history import History

if TYPE_CHECKING:
    from aisuite.utils.tools import ToolCall


TOKEN = 'test-secret-never-print'


def call(name: str = 'read', arguments: str = '{"path":"f"}', call_id: str = 'call_1') -> ResponseFunctionToolCallParam:
    return {'type': 'function_call', 'id': 'fc_' + call_id, 'call_id': call_id,
            'name': name, 'arguments': arguments, 'status': 'completed'}


def reasoning() -> ResponseReasoningItemParam:
    return {'type': 'reasoning', 'id': 'rs_1', 'summary': [],
            'encrypted_content': 'encrypted-reasoning-test'}


def message(text: str = 'Done', refusal: bool = False) -> ResponseOutputMessageParam:
    content: ResponseOutputRefusalParam | ResponseOutputTextParam = ({'type': 'refusal', 'refusal': text} if refusal else
               {'type': 'output_text', 'text': text, 'annotations': [], 'logprobs': []})
    return {'type': 'message', 'id': 'msg_1', 'role': 'assistant',
            'status': 'completed', 'content': [content]}


def completed(output: Sequence[OutputItem], status: str = 'completed') -> WireCompleted:
    return {'type': 'response.' + status, 'sequence_number': 2,
            'response': {'id': 'resp_test', 'object': 'response', 'created_at': 0,
                         'model': 'selected-model', 'status': status,
                         'parallel_tool_calls': False, 'tool_choice': 'auto',
                         'tools': [], 'output': output}}


def item_done(item: OutputItem, index: int = 0) -> WireItemDone:
    return {'type': 'response.output_item.done', 'sequence_number': 1,
            'output_index': index, 'item': item}


def delta(text: str = 'Done', refusal: bool = False) -> WireDelta:
    return {'type': 'response.refusal.delta' if refusal else 'response.output_text.delta',
            'delta': text, 'item_id': 'msg_1', 'output_index': 0,
            'content_index': 0, 'sequence_number': 1, 'logprobs': []}


def sse(events: Sequence[WireEvent]) -> bytes:
    return ''.join('event: ' + item['type'] + '\ndata: ' + json.dumps(item) + '\n\n'
                   for item in events).encode()


class AgentLoopHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cancel = Event()
        self.events = ReplyEvents()
        self.requests: list[RequestSnapshot] = []
        self.executions: list[ToolCall] = []
        self.clients: list[httpx2.Client] = []
        self.credentials = fixture_credentials(TOKEN)
        self.input: History = [{'role': 'user', 'content': 'Inspect the workspace.'}]

    def run_stream(self, responses: Sequence[HTTPFixture], *, execute: Callable[[list[ToolCall]], list[str]] | None = None, compact_threshold: int | None = None, compact_only: bool = False) -> list[ReplyEvent]:
        real_openai = openai.OpenAI
        def handler(request: httpx2.Request) -> httpx2.Response:
            self.requests.append(REQUEST_ADAPTER.validate_json(request.content))
            index = len(self.requests)-1
            self.assertLess(index, len(responses), 'Unexpected extra model request.')
            fixture = responses[index]
            if isinstance(fixture, tuple):
                status, body = fixture
                return httpx2.Response(status, json=body)
            return httpx2.Response(200, content=sse(fixture), headers={'content-type': 'text/event-stream'})
        def create_client(*, api_key: str, base_url: str, timeout: float, max_retries: int) -> openai.OpenAI:
            client = httpx2.Client(transport=httpx2.MockTransport(handler))
            self.clients.append(client)
            return real_openai(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, http_client=client)
        def tool_execute(tool_calls: ToolCall | list[ToolCall]) -> list[str]:
            entries = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            self.executions.extend(deepcopy(entries))
            if execute:
                return execute(entries)
            return ['tool result'] * len(entries)
        output = io.StringIO()
        with (patch('openai.OpenAI', side_effect=create_client),
              patch.object(Tools, 'execute', side_effect=tool_execute),
              redirect_stdout(output), redirect_stderr(output)):
            stream_chatgpt_response(self.credentials, 'selected-model', list(self.input),
                                   self.events, self.cancel, compact_threshold=compact_threshold,
                                   compact_only=compact_only)
        self.assertNotIn(TOKEN, output.getvalue())
        self.assertTrue(all(client.is_closed for client in self.clients), 'Close every SDK/HTTP client.')
        result: list[ReplyEvent] = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
        self.assertNotIn(TOKEN, str(result))
        return result

    def assert_subset(self, expected: Mapping[str, object], actual: Mapping[str, JSONValue]) -> None:
        for key, value in expected.items():
            self.assertEqual(actual.get(key), value)

    def test_selected_model_medium_reasoning_and_non_strict_function_schemas(self) -> None:
        events = self.run_stream([[delta(), completed([message()])]])
        self.assertIn(('text', 'Done'), events)
        self.assertTrue(any(kind == 'complete' for kind, _ in events))
        request, = self.requests
        self.assertEqual(request['model'], 'selected-model')
        self.assertEqual(request.get('reasoning'), {'effort': 'medium'})
        self.assertIs(request['store'], False)
        self.assertNotIn('context_management', request)
        self.assertIn('reasoning.encrypted_content', request['include'])
        schemas = request['tools']
        self.assertEqual({item['name'] for item in schemas}, {'read', 'write', 'edit', 'bash', 'task'})
        for schema in schemas:
            self.assertEqual(schema['type'], 'function')
            self.assertIs(schema['strict'], False)
            self.assertNotIn('function', schema)
        read = next(item for item in schemas if item['name'] == 'read')
        parameters = read['parameters']
        assert parameters is not None
        required = parameters.get('required', [])
        assert is_string_list(required)
        self.assertNotIn('offset', required)
        self.assertNotIn('limit', required)
        self.assertEqual(self.input, [{'role': 'user', 'content': 'Inspect the workspace.'}])

    def test_subagent_request_uses_its_own_prompt_tools_and_input(self) -> None:
        arguments = '{"description":"Find sessions","prompt":"Find where sessions are saved."}'
        events = self.run_stream([
            [completed([call('task', arguments, 'call_task')])],
            [delta('Found it'), completed([message('Found it')])],
            [delta(), completed([message()])],
        ])

        parent, child, resumed = self.requests
        self.assertEqual(parent['instructions'], prompts.SYSTEM_PROMPT + prompts.TASK_PROMPT)
        self.assertEqual(child['instructions'], prompts.SYSTEM_PROMPT + prompts.SUBAGENT_PROMPT)
        self.assertIn('task', {item['name'] for item in parent['tools']})
        self.assertEqual({item['name'] for item in child['tools']}, {'read', 'write', 'edit', 'bash'})
        self.assertEqual(child['input'], [{'role': 'user', 'content': 'Find where sessions are saved.'}])
        self.assert_subset({'type': 'function_call_output', 'call_id': 'call_task', 'output': 'Found it'},
                           resumed['input'][-1])
        self.assertNotIn(('text', 'Found it'), events)
        self.assertIn(('subagent', {'task_id': 'call_task', 'event': ('text', 'Found it')}), events)
        started = next(value for kind, value in events if kind == 'tool_started')
        self.assertEqual(started, {'name': 'task', 'detail': 'Find sessions', 'task_id': 'call_task',
                                   'prompt': 'Find where sessions are saved.'})
        self.assertIn(('text', 'Done'), events)
        self.assertEqual(self.executions, [])

    def test_no_request_uses_server_side_context_management(self) -> None:
        events = self.run_stream([
            [completed([call()])],
            [delta(), completed([message()])],
        ], compact_threshold=400000)

        self.assertEqual(len(self.requests), 2)
        for request in self.requests:
            self.assertNotIn('context_management', request)
            self.assertNotEqual(request['instructions'], prompts.COMPACTION_PROMPT)
        self.assertEqual(events[-1][0], 'complete')

    def test_manual_mode_sends_a_toolless_summary_request_and_keeps_only_the_summary(self) -> None:
        original = deepcopy(self.input)
        events = self.run_stream([[delta('Goal: inspect.'), completed([message('Goal: inspect.')])]],
                                 compact_only=True)

        request, = self.requests
        self.assertEqual(request['tools'], [])
        self.assertEqual(request['input'], original)
        self.assertEqual(request['instructions'], prompts.COMPACTION_PROMPT)
        self.assertNotIn('context_management', request)
        self.assertEqual(events, [('complete', [
            {'role': 'user', 'content': agent.SUMMARY_PREFIX + 'Goal: inspect.'}])])
        self.assertEqual(self.executions, [])
        self.assertEqual(self.input, original)

    def test_manual_summary_without_text_fails_and_keeps_context(self) -> None:
        original = deepcopy(self.input)
        events = self.run_stream([[item_done(reasoning()), completed([reasoning()])]], compact_only=True)

        self.assertEqual(events, [('error', 'Compaction returned no summary. Please try again.')])
        self.assertEqual(self.input, original)

    def test_replays_reasoning_every_output_and_matching_tool_result(self) -> None:
        outputs: list[OutputItem] = [reasoning(), message('Checking.'), call()]
        events = self.run_stream([[delta('Checking.'), completed(outputs)],
                                  [delta('Done'), completed([message()])]])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.executions), 1)
        replay = self.requests[1]['input']
        self.assertEqual(replay[0], self.input[0])
        for expected, actual in zip(outputs, replay[1:4], strict=True):
            self.assert_subset(expected, actual)
        result = replay[4]
        self.assertEqual(result['type'], 'function_call_output')
        self.assertEqual(result['call_id'], 'call_1')
        self.assertIn('tool result', str(result['output']))
        self.assertTrue(any(kind == 'complete' for kind, _ in events))

        self.assertIn(('tool_started', {'name': 'read', 'detail': 'f'}), events)
        self.assertIn(('tool_finished', 'read'), events)

    def test_existing_checkpoint_does_not_prune_without_new_compaction_output(self) -> None:
        self.input = [
            {'role': 'user', 'content': 'Retained user input before a saved checkpoint'},
            {'type': 'compaction', 'id': 'cmp_saved', 'encrypted_content': 'fake-saved-context'},
            {'role': 'user', 'content': 'Continue the task'},
        ]
        original = deepcopy(self.input)
        events = self.run_stream([[delta(), completed([message()])]])

        complete = next(event[1] for event in events if event[0] == 'complete')
        assert complete is not None
        self.assertEqual(complete[:len(original)], original)
        self.assertEqual(self.input, original)

    def test_each_completed_response_reports_its_own_usage_before_followup_work(self) -> None:
        first = completed([call()])
        first['response']['usage'] = {
            'input_tokens': 1200,
            'input_tokens_details': {'cached_tokens': 800, 'cache_write_tokens': 0},
            'output_tokens': 100,
            'output_tokens_details': {'reasoning_tokens': 40},
            'total_tokens': 1300,
        }
        second = completed([message()])
        second['response']['usage'] = {
            'input_tokens': 1500,
            'input_tokens_details': {'cached_tokens': 1000, 'cache_write_tokens': 0},
            'output_tokens': 50,
            'output_tokens_details': {'reasoning_tokens': 20},
            'total_tokens': 1550,
        }

        events = self.run_stream([[first], [second]])

        self.assertEqual([value for kind, value in events if kind == 'usage'], [
            {'input_tokens': 1200, 'output_tokens': 100, 'total_tokens': 1300},
            {'input_tokens': 1500, 'output_tokens': 50, 'total_tokens': 1550},
        ])
        kinds = [kind for kind, _ in events]
        usage_indices = [index for index, kind in enumerate(kinds) if kind == 'usage']
        self.assertLess(usage_indices[0], kinds.index('tool_started'))
        self.assertLess(usage_indices[1], kinds.index('complete'))
        self.assertEqual(len(self.executions), 1)
        self.assertEqual(kinds[-1], 'complete')

    def test_unsuccessful_or_inconsistent_responses_do_not_report_usage(self) -> None:
        fixtures: list[list[WireEvent]] = [
            [completed([message()], 'failed')],
            [completed([message()], 'incomplete')],
            [delta('Different preview'), completed([message()])],
        ]
        for fixture in fixtures:
            with self.subTest(ending=fixture[-1]['type'], preview=len(fixture) > 1):
                self.requests.clear()
                ending = fixture[-1]
                assert 'response' in ending
                ending['response']['usage'] = {
                    'input_tokens': 1200,
                    'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0},
                    'output_tokens': 100,
                    'output_tokens_details': {'reasoning_tokens': 40},
                    'total_tokens': 1300,
                }
                events = self.run_stream([fixture])
                self.assertNotIn('usage', [kind for kind, _ in events])
                self.assertEqual(events[-1][0], 'error')

    def test_item_done_output_survives_empty_completion_and_replays_with_tool_result(self) -> None:
        outputs: list[OutputItem] = [reasoning(), call(arguments='{"path":"README.md"}')]
        events = self.run_stream([
            [*(item_done(item, index) for index, item in enumerate(outputs)), completed([])],
            [item_done(message()), completed([])],
        ])
        self.assertEqual(len(self.executions), 1)
        self.assertEqual(len(self.requests), 2)
        replay = self.requests[1]['input']
        self.assert_subset(outputs[0], replay[1])
        self.assert_subset(outputs[1], replay[2])
        self.assertEqual(replay[3], {'type': 'function_call_output',
                                  'call_id': 'call_1', 'output': 'tool result'})
        self.assertIn(('text', 'Done'), events)
        self.assertEqual(events[-1][0], 'complete')

    def test_item_done_calls_do_not_execute_without_successful_response_completion(self) -> None:
        for ending in ([], [completed([], 'incomplete')], [completed([], 'failed')]):
            with self.subTest(ending=ending):
                self.requests.clear()
                self.executions.clear()
                events = self.run_stream([[item_done(reasoning()), item_done(call(), 1), *ending]])
                self.assertEqual(self.executions, [])
                self.assertEqual(events[-1][0], 'error')

    def test_completed_arguments_win_over_partial_argument_deltas(self) -> None:
        partial: WireArgumentsDelta = {'type': 'response.function_call_arguments.delta', 'delta': '{"path":"WRONG"}',
                   'item_id': 'fc_call_1', 'output_index': 0, 'sequence_number': 1}
        events = self.run_stream([[partial, completed([call(arguments='{"path":"correct"}')])],
                                  [delta(), completed([message()])]])
        self.assertEqual(len(self.executions), 1)
        item = self.executions[0]
        arguments = item['function']['arguments']
        self.assertEqual(json_value(cast(object, json.loads(arguments))),
                         {'path': 'correct'})
        self.assertIn(('tool_started', {'name': 'read', 'detail': 'correct'}), events)

    def test_tool_exception_is_returned_to_model_for_recovery(self) -> None:
        def broken(entries: list[ToolCall]) -> list[str]:
            raise ValueError('Missing file')
        events = self.run_stream([[completed([call()])], [delta(), completed([message()])]], execute=broken)
        self.assertEqual(len(self.requests), 2)
        result = self.requests[1]['input'][-1]
        self.assertEqual(result['call_id'], 'call_1')
        self.assertRegex(str(result['output']).lower(), r'error|fail|missing')
        self.assertTrue(any(kind == 'complete' for kind, _ in events))
        self.assertIn(('tool_started', {'name': 'read', 'detail': 'f'}), events)
        self.assertIn(('tool_failed', 'read'), events)

    def test_nonzero_bash_exit_reaches_model_unchanged_and_allows_a_final_answer(self) -> None:
        tool_output = 'Exit code: 7\ncommand failed'
        events = self.run_stream([
            [completed([call('bash', '{"command":"exit 7"}')])],
            [delta('The command exited with code 7.'), completed([message('The command exited with code 7.')])],
        ], execute=lambda entries: [tool_output])
        self.assertIn(('tool_failed', 'bash'), events)
        self.assertNotIn(('tool_finished', 'bash'), events)
        result = self.requests[1]['input'][-1]
        self.assertEqual(result, {'type': 'function_call_output', 'call_id': 'call_1',
                                  'output': tool_output})
        self.assertEqual(events[-1][0], 'complete')

    def test_incomplete_failed_or_eof_never_executes_a_tool(self) -> None:
        partial: WireArgumentsDelta = {'type': 'response.function_call_arguments.delta', 'delta': '{"path":"f"}',
                   'item_id': 'fc_call_1', 'output_index': 0, 'sequence_number': 1}
        fixtures: list[list[WireEvent]] = [[], [partial], [completed([call()], 'incomplete')],
                        [completed([call()], 'failed')],
                        [completed([{**call(), 'status': 'in_progress'}])]]
        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                self.requests.clear()
                events = self.run_stream([fixture])
                self.assertEqual(self.executions, [])
                self.assertFalse(any(kind == 'complete' for kind, _ in events))
                self.assertTrue(any(kind == 'error' for kind, _ in events))
                diagnostic = next(event[1] for event in events if event[0] == 'error')
                self.assertRegex(diagnostic.lower(), r'complet')

    def test_refusal_deltas_remain_streamed(self) -> None:
        events = self.run_stream([[delta('Cannot help.', refusal=True),
                                  completed([message('Cannot help.', refusal=True)])]])
        self.assertIn(('text', 'Cannot help.'), events)
        self.assertTrue(any(kind == 'complete' for kind, _ in events))
        self.assertEqual(self.executions, [])

    def test_calls_past_eight_are_answered_as_skipped_and_the_turn_continues(self) -> None:
        events = self.run_stream([[completed([call(call_id=f'call_{n}') for n in range(10)])],
                                  [delta(), completed([message()])]])
        self.assertEqual(len(self.executions), 8)
        outputs = {item['call_id']: item['output'] for item in self.requests[1]['input']
                   if item.get('type') == 'function_call_output'}
        self.assertEqual(len(outputs), 10)
        self.assertEqual(outputs['call_8'], agent.SKIPPED_CALL)
        self.assertEqual(outputs['call_9'], agent.SKIPPED_CALL)
        self.assertEqual(events[-1][0], 'complete')

    def test_the_sixteenth_request_asks_for_a_summary_and_never_runs_tools(self) -> None:
        events = self.run_stream([[completed([call(call_id=f'call_{n}')])] for n in range(16)])
        self.assertEqual(len(self.requests), 16)
        self.assertEqual(len(self.executions), 15)
        last_input = self.requests[-1]['input']
        self.assertEqual(last_input[-1], {'role': 'user', 'content': agent.FINAL_REQUEST_NOTE})
        self.assertNotIn(agent.FINAL_REQUEST_NOTE, str(self.requests[-2]['input']))
        kind, history = events[-1]
        self.assertEqual(kind, 'complete')
        assert isinstance(history, list)
        self.assertEqual(history[-1], {'type': 'function_call_output', 'call_id': 'call_15',
                                       'output': agent.BUDGET_USED})
        self.assertNotIn('error', [kind for kind, _ in events])

    def test_cancel_after_first_tool_stops_remaining_calls_and_requests(self) -> None:
        def cancel_after_one(entries: list[ToolCall]) -> list[str]:
            self.assertEqual(len(entries), 1, 'Execute sequentially so cancellation can stop the next tool.')
            self.cancel.set()
            return ['write finished']
        events = self.run_stream([[completed([call('write', '{"path":"f","content":"ok"}'),
                                              call(call_id='call_2')])]], execute=cancel_after_one)
        self.assertEqual(len(self.executions), 1)
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(any(kind == 'complete' for kind, _ in events))

    def test_api_diagnostics_include_status_code_param_without_raw_error(self) -> None:
        events = self.run_stream([(400, {'error': {'message': TOKEN,
                                    'type': 'invalid_request_error', 'code': 'unsupported_value',
                                    'param': 'reasoning.effort'}})])
        error = next(event[1] for event in events if event[0] == 'error')
        self.assertIn('400', error)
        self.assertIn('unsupported_value', error)
        self.assertIn('reasoning.effort', error)
        self.assertEqual(self.executions, [])

    def test_cancellation_before_request_is_silent(self) -> None:
        self.cancel.set()
        self.assertEqual(self.run_stream([]), [])
        self.assertEqual(self.requests, [])
        self.assertEqual(self.executions, [])

    def test_completed_text_without_deltas_finishes_successfully(self) -> None:
        events = self.run_stream([[completed([message("Recovered")])]])
        self.assertIn(('text', 'Recovered'), events)
        self.assertTrue(any(kind == 'complete' for kind, _ in events))

    def test_empty_completion_reports_structure_without_reasoning_contents(self) -> None:
        events = self.run_stream([[completed([reasoning()])]])
        diagnostic = next(event[1] for event in events if event[0] == 'error')
        self.assertIn('completed', diagnostic)
        self.assertIn('reasoning', diagnostic)
        self.assertNotIn('encrypted-reasoning-test', diagnostic)


class AgentHistoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        def reuse_credentials(saved: Credentials) -> Credentials:
            return saved
        self.enterContext(patch("vin.tui.app.refresh_credentials", side_effect=reuse_credentials))
        self.owner = tui.VinApp()
        self.owner.credentials = fixture_credentials(TOKEN)
        self.owner.selected_model = 'selected-model'
        self.draft = Document('next draft', cursor_position=3)
        self.owner.session = cast(PromptSession[str], Mock(spec=PromptSession, app=Mock(), default_buffer=Buffer(document=self.draft)))
        self.used, self.release, self.closed = Event(), Event(), Event()
        self.client = ClientDouble(self.closed.set)

    @contextmanager
    def stream(self, events: Iterable[ResponseStreamEvent]) -> Generator[Iterator[ResponseStreamEvent]]:
        yield iter(events)

    async def wait_until(self, predicate: Callable[[], bool]) -> None:
        async with asyncio.timeout(2):
            while not predicate():
                await asyncio.sleep(.001)

    async def exercise(self, outcome: Literal["complete", "cancel", "error"]) -> None:
        first = ResponseCompletedEvent.model_validate(completed([reasoning(), call('write',
                    '{"path":"f","content":"written"}')]))
        final_delta = ResponseTextDeltaEvent.model_validate(delta())
        final = ResponseCompletedEvent.model_validate(completed([message()]))
        def second_response() -> Iterator[ResponseStreamEvent]:
            self.release.wait(2)
            yield final_delta
            yield final
        request = httpx2.Request('POST', 'https://api.openai.com/v1/responses')
        api_error = openai.APIStatusError(TOKEN, response=httpx2.Response(500, request=request), body=None)
        self.client.responses.create.side_effect = [self.stream([first]),
             api_error if outcome == 'error' else self.stream(second_response())]
        def execute(*args: object, **kwargs: object) -> list[str]:
            self.used.set()
            return ['Wrote file.']
        task = None
        with patch('openai.OpenAI', return_value=self.client), patch.object(Tools, 'execute', side_effect=execute):
            try:
                task = asyncio.create_task(self.owner.stream_reply('Change the file.'))
                active_task = task
                self.owner.pending_reply = task
                await self.wait_until(lambda: self.used.is_set() or active_task.done())
                self.assertTrue(self.used.is_set(), 'The completed write call must execute.')
                if outcome == 'cancel':
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                else:
                    self.release.set()
                    await asyncio.wait_for(task, 2)
                self.assertEqual(self.owner.session.default_buffer.document, self.draft)
                self.assertEqual(self.owner.messages[0], {'role': 'user', 'content': 'Change the file.'})
                if outcome == 'complete':
                    kinds = {kind for item in self.owner.messages if isinstance(kind := item.get('type'), str)}
                    self.assertTrue({'reasoning', 'function_call', 'function_call_output'} <= kinds)
                else:
                    messages = self.owner.messages
                    self.assertEqual(messages[-1]["role"], "assistant")
                    self.assertIn("Turn interrupted after tool use", str(messages[-1]["content"]))
                    self.assertIn("function_call", [item.get("type") for item in messages])
                    self.assertIn({"type": "function_call_output", "call_id": "call_1", "output": "Wrote file."},
                                  messages)
                    self.assertEqual(agent.close_open_calls(messages), messages, "Every call has a result")
                self.assertNotIn(TOKEN, str(self.owner.messages))
            finally:
                self.release.set()
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await self.wait_until(self.closed.is_set)

    async def test_success_keeps_full_response_items_in_followup_history(self) -> None:
        await self.exercise('complete')

    async def test_cancel_after_tool_use_retains_request_interruption_note_and_draft(self) -> None:
        await self.exercise('cancel')

    async def test_api_error_after_tool_use_retains_request_interruption_note_and_draft(self) -> None:
        await self.exercise('error')


if __name__ == '__main__':
    unittest.main()

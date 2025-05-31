import { useEffect, useState } from 'react';
import axios from 'axios';
import { Message, ToolOutput, PlanItem, SubtaskStatus } from '../types';

export function useChat(threadId: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [blocks, setBlocks] = useState<ToolOutput[] | null>(null);
  const [plan, setPlan] = useState<PlanItem[]>([]);
  const [subStatus, setSubStatus] = useState<SubtaskStatus[]>([]);
  const [reportUrl, setReportUrl] = useState<string | null>(null);
  const [status, setStatus] = useState('Agents ready');

  useEffect(() => {
    const interval = setInterval(() => {
      fetchLatest();
      fetchCurrentPlan();
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const fetchLatest = async () => {
    try {
      const { data } = await axios.get('/latest_answer');
      if (data.answer) {
        setMessages(prev => [...prev, { type: 'agent', content: data.answer }]);
      }
      setBlocks(data.blocks || null);
      setPlan(data.plan || []);
      setSubStatus(data.subtask_status || []);
      setReportUrl(data.final_report_url || null);
      setStatus('');
    } catch (err) {
      console.error(err);
    }
  };

  const fetchCurrentPlan = async () => {
    try {
      const res = await axios.get('/current_plan');
      setPlan(res.data.plan || []);
      setSubStatus(res.data.subtask_status || []);
      setReportUrl(res.data.final_report_url || null);
      setBlocks(res.data.blocks || null);
    } catch (err) {
      console.error(err);
    }
  };

  const sendQuery = async (q: string) => {
    setMessages(prev => [...prev, { type: 'user', content: q }]);
    await axios.post('/query', { query: q, tts_enabled: false });
  };

  const stop = async () => {
    await axios.get('/stop');
  };

  return { messages, status, sendQuery, stop, blocks, plan, subStatus, reportUrl };
}

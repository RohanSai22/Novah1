import { useParams } from 'react-router-dom';
import { useChat } from '../hooks/useChat';
import ChatArea from '../components/ChatArea';
import PromptInput from '../components/PromptInput';
import Sidebar from '../components/Sidebar';
import AgentWorkspace from '../components/AgentWorkspace';
import { useState } from 'react';
import { Thread } from '../types';
import PlanList from '../components/PlanList';
import FinalReportCard from '../components/FinalReportCard';

export default function Chat() {
  const { threadId = '' } = useParams();
  const { messages, sendQuery, blocks, status, plan, subStatus, reportUrl } = useChat(threadId);
  const [threads] = useState<Thread[]>([{ id: threadId, messages, blocks }]);

  const handleSubmit = (prompt: string) => {
    sendQuery(prompt);
  };

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar threads={threads} />
      <div className="flex flex-col flex-1">
        <div className="border-b border-white/20 px-4 py-2">{status}</div>
        <PlanList plan={plan} status={subStatus} />
        <ChatArea messages={messages} />
        <FinalReportCard url={reportUrl} />
        <div className="p-4">
          <PromptInput onSubmit={prompt => handleSubmit(prompt)} />
        </div>
      </div>
      <AgentWorkspace blocks={blocks} />
    </div>
  );
}

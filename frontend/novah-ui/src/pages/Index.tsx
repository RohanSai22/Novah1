import { Route, Routes } from 'react-router-dom';
import Home from './Home';
import Chat from './Chat';

export default function Index() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/chat/:threadId" element={<Chat />} />
    </Routes>
  );
}

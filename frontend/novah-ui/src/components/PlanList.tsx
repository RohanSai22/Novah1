import { motion } from 'framer-motion';

interface PlanItem {
  task: string;
  tool: string;
  subtasks: string[];
}

interface Props {
  plan: PlanItem[];
  status: { subtask: string; status: string }[];
}

export default function PlanList({ plan, status }: Props) {
  const statuses = Object.fromEntries(status.map((s, i) => [i, s.status]));
  let idx = 0;
  return (
    <div className="p-4 space-y-2">
      {plan.map(p => (
        <div key={p.task} className="bg-white/10 rounded-xl p-2">
          <div className="font-semibold">{p.task} – {p.tool}</div>
          <ul className="ml-4 list-disc list-inside">
            {p.subtasks.map(st => {
              const state = statuses[idx++] || 'pending';
              return (
                <motion.li
                  key={st}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex items-center gap-2"
                >
                  <span>{state === 'done' ? '✔' : state === 'failed' ? '✖' : '○'}</span>
                  <span>{st}</span>
                </motion.li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}

import DatasetCollector from '@/components/DatasetCollector';

// Dev-only route for collecting the self-trained gesture classifier's
// training data (see ml/train.py). Not linked from the production UI.
export default function CollectPage() {
  return <DatasetCollector />;
}

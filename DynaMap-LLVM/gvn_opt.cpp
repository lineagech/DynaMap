#include <llvm/Pass.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h> // for DI...
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/Analysis/BasicAliasAnalysis.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Analysis/DomTreeUpdater.h>
#include <llvm/Analysis/MemoryDependenceAnalysis.h>
#include <llvm/Analysis/GlobalsModRef.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Config/llvm-config.h>

#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>

using namespace llvm;

namespace llvm {

class GvnOptPass : public FunctionPass {
public:
    static char ID;
    bool init = false;
    Module* module;
    DataLayout* data_layout;
    
    std::unordered_map<Value*, bool> is_val_cnst_map;
    std::unordered_map<Value*, Value*> val_cnst_map;
    
    std::unordered_set<std::string> kernel_set;

    explicit GvnOptPass() : FunctionPass(ID) 
    {
        //initializeGvnOptPassPass(*PassRegistry::getPassRegistry()); 
    }

    bool runOnFunction(Function &F) override {
        if (!init) {
            module = F.getParent();
            data_layout = new DataLayout(module);
            collectKernels(module);
            init = true;
        }
        
        if (kernel_set.find(F.getName().str()) != kernel_set.end()) { 
            /* constant propagation */
            constant_propagate(F);
            errs() << __func__ << ": " << F.getName() << "\n";
            for (auto f_inst = inst_begin(F); f_inst != inst_end(F); f_inst++) {
                process_equal_value_number(&*f_inst); 
            }
            erase_inst_from_set();

            for (auto bb = F.begin(); bb != F.end(); bb++) {
                handle_cse(&*bb);
            }
            erase_inst_from_set();
        }
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
        //AU.addRequired<DominatorTreeWrapperPass>();
        //AU.addRequired<MemoryDependenceWrapperPass>();
        AU.addRequired<AAResultsWrapperPass>();
        //AU.addRequired<GlobalsAAWrapperPass>();
        AU.addRequired<TargetLibraryInfoWrapperPass>();
    }
    

    void constant_propagate (Function& func)
    {
        /*
        if (skipFunction(func)) {
            return;
        }
        */
        errs() << "Func: " << func.getName() << "\n";
        SmallPtrSet<Instruction*, 16> work_list;
        SmallVector<Instruction*, 16> work_list_vec;
        for (auto &inst : instructions(&func)) {
            work_list.insert(&inst);
            work_list_vec.push_back(&inst);
        }
        TargetLibraryInfo *tlb_info = &getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(func);

        while (!work_list_vec.empty()) {
            SmallVector<Instruction*, 16> next_work_list_vec;
            for (auto inst : work_list_vec) {
                if (Constant *cnst = ConstantFoldInstruction(inst, *data_layout, tlb_info)) {
                    // if (!DebugCounter::shouldExecute)
                    for (User* user : inst->users()) {
                        if (work_list.insert(cast<Instruction>(user)).second) {
                            next_work_list_vec.push_back(cast<Instruction>(user));
                        }
                    }
                    errs() << "replace consts : "; inst->dump();
                    inst->replaceAllUsesWith(cnst);
                    if (isInstructionTriviallyDead(inst, tlb_info)) {
                        inst->eraseFromParent();
                    }
                }
                else if (StoreInst* store_inst = dyn_cast<StoreInst>(inst)) {
                    inst->dump();
                    if (isa<Constant>(store_inst->getValueOperand())) {
                        is_val_cnst_map[store_inst->getPointerOperand()] = true;
                        val_cnst_map[store_inst->getPointerOperand()] = store_inst->getValueOperand();
                        errs() << "Store a constant : "; inst->dump();
                    }
                    else {
                        is_val_cnst_map[store_inst->getPointerOperand()] = false;
                    }
                }
                else if (LoadInst* load_inst = dyn_cast<LoadInst>(inst)) {
                    Value* ptr_val = load_inst->getPointerOperand();
                    if (is_val_cnst_map.find(ptr_val) != is_val_cnst_map.end() 
                            && is_val_cnst_map[ptr_val]) {
                        Value* replacing_val = val_cnst_map[ptr_val]; 
                        inst->replaceAllUsesWith(replacing_val);
                        if (isInstructionTriviallyDead(inst, tlb_info)) {
                            errs() << "Erase from parent : "; inst->dump();
                            inst->eraseFromParent();
                        }
                    }
                }
            }
            work_list_vec = std::move(next_work_list_vec);
        }
    }

    void process_equal_value_number (Value* val) 
    {
        AAResults &AA = getAnalysis<AAResultsWrapperPass>().getAAResults();
        LoadInst* load_inst = dyn_cast<LoadInst>(val);
        Value* alloc_ptr_val = NULL;
        Value* replacing_val = NULL;
        if (!load_inst) return;

        Value* load_ptr = load_inst->getPointerOperand();
        BasicBlock* curr_bb = load_inst->getParent();
        size_t load_size = data_layout->getTypeStoreSize(load_inst->getType());

        std::set<Value*> common_load_vec;
        std::set<Value*> common_store_vec;
        BasicBlock::iterator prev_inst = curr_bb->begin();
        while (&(*prev_inst) != val) prev_inst++;
        
        errs() << "Start tracing back: "; prev_inst->dump();

        for (; prev_inst != curr_bb->begin(); ) {
            prev_inst--;
            /* find the first previous load */
            if (auto prev_load_inst = dyn_cast<LoadInst>(prev_inst)) {
                if (prev_load_inst->getPointerOperand() != load_ptr) continue;
                errs() << "Found: "; prev_load_inst->dump();
                replacing_val = &*prev_inst;
            }
            if (auto prev_store_inst = dyn_cast<StoreInst>(prev_inst)) {
                if (AA.getModRefInfo(prev_store_inst, load_ptr, load_size) == ModRefInfo::Mod) {
                    if (prev_store_inst->getPointerOperand() == load_ptr)
                        return;
                }
            }
        }
        if (replacing_val == NULL) return;
        /* the candidates operand may be replced */
        //std::set<Value*> candidates;
        //DominatorTree &DT = getAnalysis<DominatorTree>();
        //for (auto use = load_inst->use_begin(); use != load_inst->use_end(); use++) {
        //    candidates.insert(*use);
        //}
        load_inst->replaceAllUsesWith(replacing_val);
        be_erased_inst_set.insert(load_inst);
    }

    void erase_inst_from_set ()
    {
        for (auto it = be_erased_inst_set.begin(), it_end = be_erased_inst_set.end(); 
                it != it_end; it++) {
            auto inst = dyn_cast<Instruction>((*it));
            errs() << "The instruction is being removed: "; inst->dump();
            inst->eraseFromParent();
        }
        be_erased_inst_set.clear();
    }

    bool is_same_expression (Value* a, Value* b)
    {
        if (isa<SExtInst>(a) && isa<SExtInst>(b)) {
            if (cast<SExtInst>(a)->getOperand(0) == cast<SExtInst>(b)->getOperand(0) 
                && cast<SExtInst>(a)->getDestTy()->getTypeID() == cast<SExtInst>(b)->getDestTy()->getTypeID()) {
                return true;
            }
            else {
                return false;
            }
        }
        else if (isa<ZExtInst>(a) && isa<ZExtInst>(b)) {
            if (cast<ZExtInst>(a)->getOperand(0) == cast<ZExtInst>(b)->getOperand(0) 
                && cast<ZExtInst>(a)->getDestTy()->getTypeID() == cast<ZExtInst>(b)->getDestTy()->getTypeID()) {
                return true;
            }
            else {
                return false;
            }
        }
        else if (isa<BinaryOperator>(a) && isa<BinaryOperator>(b)) {
            BinaryOperator *a_bin = cast<BinaryOperator>(a);
            BinaryOperator *b_bin = cast<BinaryOperator>(b);
            if (a_bin->getOpcode() != b_bin->getOpcode()) {
                return false;
            }
            Instruction::BinaryOps op_code = a_bin->getOpcode();
            if (isa<Constant>(a_bin->getOperand(0)) && isa<Constant>(a_bin->getOperand(1))) {
                ConstantInt *a_bin_op_cnst1 = cast<ConstantInt>(a_bin->getOperand(0));
                ConstantInt *b_bin_op_cnst1 = cast<ConstantInt>(b_bin->getOperand(0));
                ConstantInt *a_bin_op_cnst2 = cast<ConstantInt>(a_bin->getOperand(1));
                ConstantInt *b_bin_op_cnst2 = cast<ConstantInt>(b_bin->getOperand(1));

                int64_t a_op_cnst_val1 = a_bin_op_cnst1->getSExtValue();
                int64_t b_op_cnst_val1 = b_bin_op_cnst1->getSExtValue();
                int64_t a_op_cnst_val2 = a_bin_op_cnst2->getSExtValue();
                int64_t b_op_cnst_val2 = b_bin_op_cnst2->getSExtValue();

                if (a_op_cnst_val1 == b_op_cnst_val1 && a_op_cnst_val2 == b_op_cnst_val2) {
                    return true;
                }
                else if (a_bin->isCommutative() && a_op_cnst_val1 == b_op_cnst_val2 && a_op_cnst_val2 == b_op_cnst_val1) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else if (isa<Constant>(a_bin->getOperand(0)) && !isa<Constant>(a_bin->getOperand(1))) {
                ConstantInt *a_bin_op_cnst = cast<ConstantInt>(a_bin->getOperand(0));
                ConstantInt *b_bin_op_cnst = cast<ConstantInt>(b_bin->getOperand(0));
                int64_t a_op_cnst_val = a_bin_op_cnst->getSExtValue();
                int64_t b_op_cnst_val = b_bin_op_cnst->getSExtValue();
                if (a_op_cnst_val == b_op_cnst_val && a_bin->getOperand(1) == b_bin->getOperand(1)) {
                    return true;
                }
                else {
                    return false;
                }
            }
            else if (!isa<Constant>(a_bin->getOperand(0)) && isa<Constant>(a_bin->getOperand(1))) {
                ConstantInt *a_bin_op_cnst = cast<ConstantInt>(a_bin->getOperand(1));
                ConstantInt *b_bin_op_cnst = cast<ConstantInt>(b_bin->getOperand(1));
                int64_t a_op_cnst_val = a_bin_op_cnst->getSExtValue();
                int64_t b_op_cnst_val = b_bin_op_cnst->getSExtValue();
                if (a_op_cnst_val == b_op_cnst_val && a_bin->getOperand(0) == b_bin->getOperand(0)) {
                    return true;
                }
                else {
                    return false;
                }

            }
            if (a_bin->getOperand(0) != b_bin->getOperand(0) 
                || a_bin->getOperand(1) != b_bin->getOperand(1)) {
                return false;
            }
            else if (a_bin->isCommutative() && (a_bin->getOperand(0) != b_bin->getOperand(1) 
                || a_bin->getOperand(1) != b_bin->getOperand(0))) {
                return false;
            }
            return true;
        }
        return false;
    }

    void handle_cse (BasicBlock* bb)
    {
        for (auto inst = bb->begin(); inst != bb->end(); inst++) {
            if (be_erased_inst_set.find(&*inst) != be_erased_inst_set.end()) {
                continue;
            }
            auto n_inst = inst;
            n_inst++;
            for ( ; n_inst != bb->end(); n_inst++) {
                if (is_same_expression(&*inst, &*n_inst)) {
                    // duplicate, eliminate it
                    errs() << "Same Exp: \n";
                    errs() <<"\t"; inst->dump();
                    errs() <<"\t"; n_inst->dump();
                    /* first replace all use */
                    n_inst->replaceAllUsesWith(&*inst);
                    /* then be deleted afterwards*/
                    be_erased_inst_set.insert(&*n_inst);
                }
            }
        }
    }

    bool collectKernels(Module* module) 
    {
        bool kernel_found = false;
        for (auto named_meta = module->named_metadata_begin(); 
                named_meta != module->named_metadata_end(); 
                named_meta++) {
            for (auto op = named_meta->op_begin(); op != named_meta->op_end(); op++) {
                MDNode* node = (*op);
                Metadata* prev_meta = NULL;
                for (auto sub_op = node->op_begin(); sub_op != node->op_end(); sub_op++) {
                    Metadata* meta = sub_op->get();
                    if (!meta) continue;
                    MDString* meta_str = dyn_cast<MDString>(meta);

                    if (meta_str && meta_str->getString().str() == "kernel") {
                        kernel_found = true;
                        std::string md_str;
                        raw_string_ostream md_str_stream(md_str);
                        prev_meta->print(md_str_stream);

                        std::string kernel_name = md_str.substr(md_str.find("@")+1);

                        if (kernel_name.find("psu_cuc1057_channel_device_init") == std::string::npos) {
                            errs() << "\t Found a kernel function: "
                                << kernel_name << "\n";
                            kernel_set.insert(kernel_name);
                        }
                    }
                    prev_meta = meta;
                }
            }
        }
        return kernel_found;
    }

private:
    std::set<Value*> be_erased_inst_set;
};

}

char GvnOptPass::ID = 0;

static RegisterPass<GvnOptPass> X("GvnOpt", "GVN Optimization for CUDA", false, false);
static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [](const PassManagerBuilder &Builder, legacy::PassManagerBase &PM) {
        PM.add(new GvnOptPass());
    }
);

/*
INITIALIZE_PASS_BEGIN(GvnOptPass, "gvn_opt", "GVN Optimization for CUDA", false, false)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(GvnOptPass, "gvn_opt", "GVN Optimization for CUDA", false, false)
*/

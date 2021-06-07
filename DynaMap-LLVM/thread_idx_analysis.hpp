#ifndef __THREAD_IDX_ANALYSIS_HPP__
#define __THREAD_IDX_ANALYSIS_HPP__
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
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/CodeGen/RegAllocRegistry.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/MemoryLocation.h>

#include <queue>
#include <vector>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <cstdarg>

#include "inc.def"

using namespace llvm;
using namespace std;

typedef enum {
    ACC_FULL_THREAD_ID = 0,
    ACC_THREAD_ID_X,
    ACC_BLOCK_ID_X,
    ACC_THREAD_ID_Y,
    ACC_BLOCK_ID_Y,
    ACC_CNST,
    ACC_NOT_FOUND,
    ACC_TYPE_NUM
} THREAD_ACCESS_TYPE_E;

string THREAD_ACCESS_TYPE_STR[ACC_TYPE_NUM] = {
    "ACC_FULL_THREAD_ID",
    "ACC_THREAD_ID_X",
    "ACC_BLOCK_ID_X",
    "ACC_THREAD_ID_Y",
    "ACC_BLOCK_ID_Y",
    "ACC_CNST",
    "ACC_NOT_FOUND"
};

class ThreadIdxAnalysis
{
private:
    std::unordered_map<Value*,bool> compose_thread_id_tracking_map;
    std::unordered_map<Value*,bool> compose_block_id_tracking_map;
    std::unordered_map<Value*,bool> is_tid_map;
    
    /* store-load use map */
    std::unordered_map<Function*, std::unordered_map<Value*, Value*> > store_load_use_map;

public:
    typedef bool (ThreadIdxAnalysis::*strip_helper)(Value*, 
                                                    vector<pair<llvm::Instruction::BinaryOps,Value*>>&, 
                                                    vector<pair<llvm::Instruction::BinaryOps,Value*>>& );
    typedef bool (ThreadIdxAnalysis::*strip_handle_bin_helper)(BinaryOperator*, 
                                                               Value*, 
                                                               vector<pair<llvm::Instruction::BinaryOps,Value*>>&, 
                                                               vector<pair<llvm::Instruction::BinaryOps,Value*>>& );
    typedef bool (ThreadIdxAnalysis::*strip_handle_bin_cond)(Value*);


 
    ThreadIdxAnalysis() {}

    Value* get_final_scale_1d (LLVMContext& context,
                               IRBuilder<>& builder,
                               vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale)
    {
        if (scale.size() == 0) {
            IntegerType* type = IntegerType::get(context, 32);
            ConstantInt* scale_cnst = ConstantInt::get(type, 1, false); 
            return dyn_cast<Value>(scale_cnst);
        }
        if (scale.size() == 1) {
            return scale[0].second;
        }
        assert(scale.size() >= 2);
        Value *bin_op = builder.CreateBinOp(scale[1].first, scale[0].second, scale[1].second);
        for (int i = 2; i < scale.size(); i++) {
            //auto next_bin_op = BinaryOperator::Create(scale[i].first, dyn_cast<Value>(bin_op), scale[i].second);
            //next_bin_op->insertAfter(bin_op);
            auto next_bin_op = builder.CreateBinOp(scale[i].first, dyn_cast<Value>(bin_op), scale[i].second);
            bin_op = next_bin_op;
        }
        return bin_op;
    }
    
    bool isBackUpdate(LoadInst* load_inst, Value* store_val) 
    {
        auto inst_it = load_inst->getParent()->begin();
        for (; &*inst_it != load_inst; inst_it++) {
            if (&*inst_it == store_val) {
                return false;
            }
        }
        for (; inst_it != load_inst->getParent()->end(); inst_it++) {
            if (&*inst_it == store_val) {
                return true;
            }
        }

        return false;
    }

    void buildLoadStoreUse(Function* func)
    {
        if (!func) return;
        unordered_map<Value*, StoreInst*> alloca_store_val_map;
        store_load_use_map.clear(); 
        //errs() << "Func Name: " << func->getName() << "\n";
        for (auto inst = inst_begin(func); inst != inst_end(func); inst++) {
            //errs() << __func__ << "=== "; inst->dump();
            if (isa<AllocaInst>(&*inst)) {
                AllocaInst *alloc_inst = dyn_cast<AllocaInst>(&*inst);
                if (alloc_inst->isArrayAllocation()) continue; 
                StoreInst* store_inst = NULL;
                for (auto user : alloc_inst->users()) {
                    if (isa<StoreInst>(user)) {
                        store_inst = dyn_cast<StoreInst>(user);
                        alloca_store_val_map.insert({store_inst->getPointerOperand(), dyn_cast<StoreInst>(user)}); 
                        errs() << "User Store Inst : "; store_inst->dump();
                    }
                    
                }
            }
            else if (isa<LoadInst>(&*inst)) {
                LoadInst* load_inst = dyn_cast<LoadInst>(&*inst);
                if (alloca_store_val_map.find(load_inst->getPointerOperand()) != alloca_store_val_map.end()) {
                    StoreInst* store_inst = alloca_store_val_map[load_inst->getPointerOperand()];
                    assert(load_inst->getPointerOperand() == store_inst->getPointerOperand());
                    if (store_load_use_map.find(func) == store_load_use_map.end()) 
                        store_load_use_map[func] = unordered_map<Value*, Value*>();
                    if (!isBackUpdate(load_inst, store_inst->getValueOperand())) {
                        store_load_use_map[func].insert({load_inst, store_inst->getValueOperand()});
                    }
                    errs() << __func__ << " => Load Inst : "; load_inst->dump(); store_inst->dump(); store_inst->getValueOperand()->dump();
                }
            }
        }
    }

    Value* getRealAddr(Value* val)
    {
        queue<BasicBlock*> bfs_q;
        unordered_set<BasicBlock*> visited;
        BasicBlock* bb = dyn_cast<Instruction>(val)->getParent();
        
        if (store_load_use_map[dyn_cast<Instruction>(val)->getParent()->getParent()].find(val) 
            != store_load_use_map[dyn_cast<Instruction>(val)->getParent()->getParent()].end()) {
            return store_load_use_map[dyn_cast<Instruction>(val)->getParent()->getParent()][val];
        }
        errs() << "getRealAddr:\t"; val->dump();
        /* process current instruction */
        for (auto inst = bb->begin(); inst != bb->end(); inst++) {
            if (&*inst == val) 
                break;
            else if (isa<StoreInst>(&*inst)) {
                StoreInst* store_inst = dyn_cast<StoreInst>(&*inst);
                if (store_inst->getPointerOperand() == val) {
                    errs() << __func__ << ": "; store_inst->dump();
                    return store_inst->getValueOperand();
                }
            }
        }

        bfs_q.push(bb);
        visited.insert(bb);
        while (!bfs_q.empty()) {
            int q_size = bfs_q.size();
            for (int i = 0; i < q_size; i++) {
                BasicBlock* curr_bb = bfs_q.front();
                bfs_q.pop();
                for (auto pred_bb : predecessors(curr_bb)) {
                    for (auto rev_inst = pred_bb->rbegin(); rev_inst != pred_bb->rend(); rev_inst++) {
                        if (isa<StoreInst>(&*rev_inst)) {
                            StoreInst* store_inst = dyn_cast<StoreInst>(&*rev_inst);
                            errs() << __func__ << ": "; store_inst->dump();
                            if (store_inst->getPointerOperand() == val) {
                                return store_inst->getValueOperand();
                            }
                        }
                    }
                    bfs_q.push(&*pred_bb);
                }
            }
        }
        return NULL;
    }
    
    void push_all_mul_val(Value* op, vector<Value*>& accumu_vec, vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale_vec)
    {
        for (auto accumu_val : accumu_vec) {
            scale_vec.push_back({llvm::Instruction::Mul, accumu_val});
        }
        if (op) {
            scale_vec.push_back({llvm::Instruction::Mul, op});
        }
    }

    THREAD_ACCESS_TYPE_E analyze_gep_for_global_2d(GetElementPtrInst* gep_inst, 
                                                   vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale_block_x,
                                                   vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale_block_y) 
    {
        if (gep_inst == NULL) {
            return ACC_NOT_FOUND;
        }
        
        //buildLoadStoreUse(gep_inst->getParent()->getParent());

        Value* ptr_val = gep_inst->getPointerOperand();
        /* now only handle one-dimension array case */
        Value* idx = gep_inst->idx_begin()->get();
        
        unordered_map<Value*, Value*> counterpart_map;
        unordered_map<Value*, unsigned> counterpart_relation;
        vector<Value*> accumu_mul_val_vec;
        vector<Value*> mul_block_x_vec;
        vector<Value*> mul_block_y_vec;
        stack<Value*> path_stack;
        stack<Value*> next_try_stack;
        THREAD_ACCESS_TYPE_E access_type = ACC_NOT_FOUND;
        
        while (idx) {
            errs() << __func__ << " ---> "; idx->dump();
            if (isa<ConstantInt>(idx)) {
                bool all_constant = true;
                for (auto subseq_idx = gep_inst->idx_begin(); subseq_idx != gep_inst->idx_end(); subseq_idx++) {
                    if (!isa<Constant>(subseq_idx->get())) {
                        all_constant = false;
                    }
                }
                if (!all_constant) {
                    return ACC_NOT_FOUND;
                }
                if (isa<LoadInst>(ptr_val)) {
                    Value* addr_ptr = dyn_cast<LoadInst>(ptr_val)->getPointerOperand();     
                    ptr_val->dump();
                    addr_ptr->dump();
                    addr_ptr = getRealAddr(ptr_val);          
                    if (isa<PHINode>(addr_ptr)) {
                        return ACC_NOT_FOUND;
                    }
                    if (isa<LoadInst>(addr_ptr)) {
                        return ACC_NOT_FOUND;
                    }
                    if (!isa<GetElementPtrInst>(addr_ptr)) {
                        return ACC_NOT_FOUND;
                    }
                    errs() << "Get Real Address -----"; addr_ptr->dump(); errs() << "------\n";
                    /* Expected this is a getelementptr instruction */
                    assert(isa<GetElementPtrInst>(addr_ptr));
                    GetElementPtrInst* gep_inst = dyn_cast<GetElementPtrInst>(addr_ptr);
                    path_stack.push(idx);
                    idx = gep_inst->idx_begin()->get();
                    errs() << "New Index --> "; idx->dump();
                    continue;
                }
                else {
                    /* we may ignore the case of constant index, too minor */
                    return ACC_CNST;
                }
            }
            else if (isa<SExtInst>(idx)) {
                SExtInst* sext_inst = dyn_cast<SExtInst>(idx);
                path_stack.push(idx);
                idx = sext_inst->getOperand(0);
            }
            else if (isa<ZExtInst>(idx)) {
                ZExtInst* zext_inst = dyn_cast<ZExtInst>(idx);
                path_stack.push(idx);
                idx = zext_inst->getOperand(0);
            }
            else if (isa<LoadInst>(idx)) {
                LoadInst* load_inst = dyn_cast<LoadInst>(idx);
                Function* func = load_inst->getParent()->getParent();
                errs() << "Func Name: " << func->getName() << "\n";
                if (store_load_use_map.find(func) != store_load_use_map.end()) {
                    errs() << "In store-load-use Map!\n";
                    if (store_load_use_map[func].find(load_inst) != store_load_use_map[func].end()) {
                        path_stack.push(idx);
                        idx = store_load_use_map[func][load_inst];
                        errs() << " idx "; idx->dump(); 
                    }
                    else {
                        //goto analyze_gep_2d_unexpected;
                        goto parse_next_try;
                    }
                }
                else {
                    //goto analyze_gep_2d_unexpected;
                    goto parse_next_try;
                }
            }
            else if (isa<BinaryOperator>(idx)) {
                BinaryOperator* bin_inst = dyn_cast<BinaryOperator>(idx);
                Value* op1 = bin_inst->getOperand(0);
                Value* op2 = bin_inst->getOperand(1);
                if (bin_inst->getOpcode() == llvm::Instruction::Mul) {
                    counterpart_map[op1] = op2;        
                    counterpart_map[op2] = op1;        
                    counterpart_relation[op1] = llvm::Instruction::Mul;        
                    counterpart_relation[op2] = llvm::Instruction::Mul;        
                    
                    if (is_block_idx_x(op1)) {
                        push_all_mul_val(op2, accumu_mul_val_vec, scale_block_x);
                        return ACC_BLOCK_ID_X;
                    }
                    if (is_block_idx_x(op2)) {
                        push_all_mul_val(op1, accumu_mul_val_vec, scale_block_x);
                        return ACC_BLOCK_ID_X;
                    }
                    /*
                    if (is_block_idx_y(op1)) {
                        push_all_mul_val(op2, accumu_mul_val_vec, scale_block_y);
                        return ACC_BLOCK_ID_Y;
                    }
                    if (is_block_idx_y(op2)) {
                        push_all_mul_val(op1, accumu_mul_val_vec, scale_block_y);
                        return ACC_BLOCK_ID_Y;
                    }
                    */
                    /* try op1 first and then op2 */
                    path_stack.push(idx);
                    idx = op1;
                    next_try_stack.push(op2);
                    accumu_mul_val_vec.push_back(op2);

                    errs() << "Mul: " << op1->getName() << " ----- counter part: " << op2->getName() << "\n";
                }
                else if (bin_inst->getOpcode() == llvm::Instruction::Add || bin_inst->getOpcode() == llvm::Instruction::Sub) {
                    BinaryOperator* bin_inst = dyn_cast<BinaryOperator>(idx);
                    Value* op1 = bin_inst->getOperand(0);
                    Value* op2 = bin_inst->getOperand(1);
                    counterpart_map[op1] = op2;        
                    counterpart_map[op2] = op1;       
                    counterpart_relation[op1] = bin_inst->getOpcode();        
                    counterpart_relation[op2] = bin_inst->getOpcode();        
                    path_stack.push(idx);
                    idx = op1;
                    next_try_stack.push(op2);
                    
                    errs() << bin_inst->getOpcode() << ": " << op1->getName() << " ----- counter part: " << op2->getName() << "\n";
                }
                else if (bin_inst->getOpcode() == llvm::Instruction::UDiv || bin_inst->getOpcode() == llvm::Instruction::SDiv
                       || bin_inst->getOpcode() == llvm::Instruction::URem || bin_inst->getOpcode() == llvm::Instruction::SRem
                       || bin_inst->getOpcode() == llvm::Instruction::Or) {
                    goto parse_next_try;    
                }
            }
            else {
                // CallInst or Argument, not list all yet
                if (is_block_idx_x(idx)) {
                    mul_block_x_vec = accumu_mul_val_vec;
                    access_type = ACC_BLOCK_ID_X;
                    errs() << "Block Idx X!\n";
                    goto out;
                }
                else if (is_block_idx_y(idx)) {
                    mul_block_y_vec = accumu_mul_val_vec;
                    access_type = ACC_BLOCK_ID_Y;
                    errs() << "Block Idx Y!\n";
                    goto out;
                }
parse_next_try: 
                //if (counterpart_map.find(idx) != counterpart_map.end()) {
                if (next_try_stack.size() > 0) {    
                    Value* next_try = next_try_stack.top();    
                    /* should back to the state */
                    while (counterpart_map[next_try] != idx) {
                        errs() << "Counter part is not idx : "; idx->dump(); 
                        errs() << "------------------------- "; next_try->dump();
                        idx = path_stack.top();
                        path_stack.pop();
                        if (isa<BinaryOperator>(idx)) {
                            auto bin_inst = dyn_cast<BinaryOperator>(idx);
                            if (bin_inst->getOpcode() == llvm::Instruction::Mul) {
                                accumu_mul_val_vec.pop_back();
                            }
                        }
                    }
                    if (counterpart_relation[next_try] == llvm::Instruction::Mul) {
                        if (accumu_mul_val_vec.back() == next_try) {
                            errs() << "pop origin idx : "; accumu_mul_val_vec.back()->dump();
                            accumu_mul_val_vec.pop_back();
                        }
                        accumu_mul_val_vec.push_back(idx);
                    }
                    idx = next_try;
                    next_try_stack.pop();
                    
                    if (next_try_stack.size() > 0 && isa<ConstantInt>(idx)) {
                        goto parse_next_try;
                    }
                }
                else {
                    errs() << "No more value ...\n";
                    idx = NULL;
                }
            }

        }

out:
        if (access_type == ACC_BLOCK_ID_X) {
            push_all_mul_val(NULL, mul_block_x_vec, scale_block_x);
            errs() << "Print all scales for block_Idx_X: \n";
            for (auto v : scale_block_x) {
                v.second->dump();
            }
        }
        else if (access_type == ACC_BLOCK_ID_Y) {
            push_all_mul_val(NULL, mul_block_y_vec, scale_block_y);
            errs() << "Print all scales for block_Idx_Y: \n";
            for (auto v : scale_block_y) {
                v.second->dump();
            }
        }

        return access_type;
        
analyze_gep_2d_unexpected:
        errs() << "Unexpected Load Inst: " << __func__ << "\n";
        assert(false); 
    }

    THREAD_ACCESS_TYPE_E analyze_gep_for_global (GetElementPtrInst* gep_inst,
                                                 vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale, 
                                                 vector<pair<llvm::Instruction::BinaryOps,Value*>>& offset)
    {
        if (gep_inst == NULL) {
            return ACC_NOT_FOUND;
        }

        Value* ptr_val = gep_inst->getPointerOperand();
        /* now only handle one-dimension array case */
        Value* idx = gep_inst->idx_begin()->get();
        
        //errs() << "\t" << __func__ << " ---> ptr: " << ptr_val->getName() << ", idx: " << idx->getName() << "\n";

        if (auto arg_gep_inst = dyn_cast<GetElementPtrInst>(ptr_val)) {
            auto acc_type = analyze_gep_for_global(arg_gep_inst, scale, offset); 
            offset.push_back({llvm::Instruction::Add, idx});
            return acc_type;
        }
        else {
            if (get_scale_offset_1d(idx, scale, offset)) {
                return ACC_FULL_THREAD_ID;
            }
            clear_all_containers(2, (void*)&scale, (void*)&offset);
            //if (get_scale_offset_2d(idx, scale, offset)) {
            //    return ACC_BLOCK_ID_Y;
            //}
        }

        return ACC_NOT_FOUND;
    }
    
    void clear_all_containers(int num_containers, ...)
    {
        va_list ap;
        va_start(ap, num_containers);
        for (int i = 2; i <= num_containers; i++) {
            void* arg_p = va_arg(ap, void*);
            vector<pair<llvm::Instruction::BinaryOps,Value*>>* vec_p = (vector<pair<llvm::Instruction::BinaryOps,Value*>>*)arg_p;
            vec_p->clear();
        }
        va_end(ap);
    }
        
    bool strip_ext_and_bin_op (Value* val, 
                               vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale, 
                               vector<pair<llvm::Instruction::BinaryOps,Value*>>& offset,
                               strip_helper helper,
                               strip_handle_bin_helper handle_bin_helper, 
                               strip_handle_bin_cond handle_bin_cond) 
    {
        bool ret = false;
        if (auto zext_inst = dyn_cast<ZExtInst>(val)) {
            val = zext_inst->getOperand(0);
            return (*this.*helper)(val, scale, offset);
        }
        else if (auto sext_inst = dyn_cast<SExtInst>(val)) {
            val = sext_inst->getOperand(0);
            return (*this.*helper)(val, scale, offset);
        }
        else if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
            Value* op1 = bin_inst->getOperand(0);
            Value* op2 = bin_inst->getOperand(1);
            if ((*this.*handle_bin_cond)(op1)) {
                ret = (*this.*helper)(op1, scale, offset);
                (*this.*handle_bin_helper)(bin_inst, op2, scale, offset);
            }
            else if ((*this.*handle_bin_cond)(op2)) {
                ret = (*this.*helper)(op2, scale, offset);
                (*this.*handle_bin_helper)(bin_inst, op1, scale, offset);
            }
            else {
                return false;
            }
        }
        return ret;
    }

    bool get_scale_offset_1d (Value* val, 
                              vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale, 
                              vector<pair<llvm::Instruction::BinaryOps,Value*>>& offset) 
    {
        int scale_int = 1;
        int offset_int = 0;
        bool ret = false;

        errs() << "===== " << __func__ << " ";
        val->dump();
        
        if (is_1d_thread_id(val)) {
            val->dump();
            return true;
        }
        
        return strip_ext_and_bin_op (val, scale, offset, 
                                     (strip_helper)&ThreadIdxAnalysis::get_scale_offset_1d,
                                     (strip_handle_bin_helper)&ThreadIdxAnalysis::handle_bin_op, 
                                     (strip_handle_bin_cond)&ThreadIdxAnalysis::is_composed_by_thread_id);
    }
    
    /* target block id y */
    bool get_scale_offset_2d (Value* val, 
                              vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale, 
                              vector<pair<llvm::Instruction::BinaryOps,Value*>>& offset) 
    {
        if (is_1d_block_id(val)) {
            val->dump();
            return true;
        }

        return strip_ext_and_bin_op (val, scale, offset, 
                                     (strip_helper)&ThreadIdxAnalysis::get_scale_offset_2d,
                                     (strip_handle_bin_helper)&ThreadIdxAnalysis::handle_bin_op, 
                                     (strip_handle_bin_cond)&ThreadIdxAnalysis::is_composed_by_block_id);
    }

    void handle_bin_op (BinaryOperator* bin_inst, 
                        Value* arg_val,
                        vector<pair<llvm::Instruction::BinaryOps,Value*>>& scale, 
                        vector<pair<llvm::Instruction::BinaryOps,Value*>>& offset) 
    {
        auto op_code = bin_inst->getOpcode();
        switch (op_code) 
        {
            case llvm::Instruction::Add :
            case llvm::Instruction::Sub :
                offset.push_back({op_code, arg_val});
                break;
            case llvm::Instruction::Mul :
            case llvm::Instruction::UDiv :
            case llvm::Instruction::SDiv :
                scale.push_back({op_code, arg_val});
                break;
             
        }
    }
    
    Value* get_primitive_val(Value* val)
    {
        if (auto alloc_val_inst = dyn_cast<AllocaInst>(val)) {
            for (auto user = val->user_begin(); user != val->user_end(); user++) {
                if (auto used_store_inst = dyn_cast<StoreInst>(*user)) { /* *user returns the pointer of user */
                    auto store_val = used_store_inst->getValueOperand();
                    val = store_val; 
                    errs() << __func__ << ": "; val->dump();
                }
            } 
        }
        if (auto zext_inst = dyn_cast<ZExtInst>(val)) {
            val = zext_inst->getOperand(0);
        }
        if (auto sext_inst = dyn_cast<SExtInst>(val)) {
            val = sext_inst->getOperand(0);
        }
        return val;
    }


    bool is_1d_block_id (Value* val) 
    {   
        bool ret = false;
        
        /* Get primitive value */
        val = get_primitive_val(val); 
        
        /* Special handling for load instruction */
        if (auto load_inst = dyn_cast<LoadInst>(val)) {
            val = load_inst->getPointerOperand();
            return is_1d_block_id(val);
        }
        
        if (is_block_idx_y(val)) {
            return true;
        }

        if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
            Value* op1 = bin_inst->getOperand(0);
            Value* op2 = bin_inst->getOperand(1);
            return is_1d_block_id(op1) | is_1d_block_id(op2); 
        }
        
        return ret;
    }

    bool is_1d_thread_id (Value* val)
    {   
        /* e.g. store i32 %add, i32* %tid */
        if (is_tid_map.find(val) != is_tid_map.end()) {
            return is_tid_map[val];
        }
        
        errs() << "===== " << __func__ << " ";
        val->dump();
        
        bool ret = false;
        /* Get primitive value */
        val = get_primitive_val(val); 
        
        /* Special handling for load instruction */
        if (auto load_inst = dyn_cast<LoadInst>(val)) {
            val = load_inst->getPointerOperand();
            return is_1d_thread_id(val);
        }

        if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
            Value* op1 = bin_inst->getOperand(0);
            Value* op2 = bin_inst->getOperand(1);
            BinaryOperator* bin_op = NULL;
            CallInst* get_tid_call_inst = NULL;
            CallInst* get_bid_call_inst = NULL;
            CallInst* get_bdim_call_inst = NULL;
            
            /* Handle extension */
            if (isa<ZExtInst>(op1)) {
                op1 = dyn_cast<ZExtInst>(op1)->getOperand(0);
            }
            else if (isa<SExtInst>(op1)) {
                op1 = dyn_cast<SExtInst>(op1)->getOperand(0);
            }
            if (isa<ZExtInst>(op2)) {
                op2 = dyn_cast<ZExtInst>(op2)->getOperand(0);
            }
            else if (isa<SExtInst>(op2)) {
                op2 = dyn_cast<SExtInst>(op2)->getOperand(0);
            }

            if (bin_op = dyn_cast<BinaryOperator>(op1)) {
                get_tid_call_inst = dyn_cast<CallInst>(op2); 
            }
            else if (bin_op = dyn_cast<BinaryOperator>(op2)) {
                get_tid_call_inst = dyn_cast<CallInst>(op1); 
            }
            else {
                return false;
            }
            //get_bid_call_inst = dyn_cast<CallInst>(bin_op->getOperand(0));
            //get_bdim_call_inst = dyn_cast<CallInst>(bin_op->getOperand(1));
            get_bid_call_inst = dyn_cast<CallInst>(get_primitive_val(bin_op->getOperand(0)));
            get_bdim_call_inst = dyn_cast<CallInst>(get_primitive_val(bin_op->getOperand(1)));

            
            if (!get_tid_call_inst || !get_bid_call_inst || !get_bdim_call_inst) return false;

            errs() << "Debug: " << __func__ << " ";
            val->dump();
            errs() << get_tid_call_inst->getCalledValue()->getName() << ",\n\t"
            << get_bid_call_inst->getCalledValue()->getName() << ",\n\t"
            << get_bdim_call_inst->getCalledValue()->getName() << "\n";

            ret |= ((get_tid_call_inst->getCalledValue()->getName().str().find(THREAD_IDX_X_STR) != string::npos) 
                    && (get_bid_call_inst->getCalledValue()->getName().str().find(BLOCK_IDX_X_STR) != string::npos) 
                    && (get_bdim_call_inst->getCalledValue()->getName().str().find(BLOCK_DIM_X_STR) != string::npos));
            
            ret |= ((get_tid_call_inst->getCalledValue()->getName().str().find(THREAD_IDX_X_STR) != string::npos) 
                    && (get_bdim_call_inst->getCalledValue()->getName().str().find(BLOCK_IDX_X_STR) != string::npos) 
                    && (get_bid_call_inst->getCalledValue()->getName().str().find(BLOCK_DIM_X_STR) != string::npos));
            


            is_tid_map[val] = ret;

            return ret;
        }
        
        is_tid_map[val] = ret;

        return ret;
    }
    
    bool is_thread_idx_x(Value* val)
    {
        if (auto call_inst = dyn_cast<CallInst>(val)) {
            return (call_inst->getCalledValue()->getName().str().find(THREAD_IDX_X_STR) != string::npos);
        }
        return false;
    }

    bool is_thread_idx_y(Value* val)
    {
        if (auto call_inst = dyn_cast<CallInst>(val)) {
            return (call_inst->getCalledValue()->getName().str().find(THREAD_IDX_Y_STR) != string::npos);
        }
        return false;
    }
    
    bool is_block_idx_x(Value* val)
    {
        if (auto call_inst = dyn_cast<CallInst>(val)) {
            return (call_inst->getCalledValue()->getName().str().find(BLOCK_IDX_X_STR) != string::npos);
        }
        return false;
    }

    bool is_block_idx_y(Value* val)
    {
        if (auto call_inst = dyn_cast<CallInst>(val)) {
            return (call_inst->getCalledValue()->getName().str().find(BLOCK_IDX_Y_STR) != string::npos);
        }
        return false;
    }

    bool is_composed_by_thread_id (Value* val)
    {
        /* the value may be calculated by several binary operations 
           but eventually there sho*/
        if (compose_thread_id_tracking_map.find(val) != compose_thread_id_tracking_map.end()) {
            return compose_thread_id_tracking_map[val];
        }
           
        errs() << "===== " << __func__ << " ";
        val->dump();

        if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
            Value* op1 = bin_inst->getOperand(0);
            Value* op2 = bin_inst->getOperand(1);
            return is_composed_by_thread_id(op1) || is_composed_by_thread_id(op2); 
        }
        else if (auto call_inst = dyn_cast<CallInst>(val)) {
            auto called_func = call_inst->getCalledValue();
            errs() << __func__ << ": " << called_func->getName() << "\n";
            if (called_func->getName().str().find(THREAD_IDX_X_STR) != string::npos) {
                compose_thread_id_tracking_map[val] = true;
                return true;
            }
        }
        else if (is_1d_thread_id(val)) {
            compose_thread_id_tracking_map[val] = true;
            errs() << __func__ << ": is thread is --> "; val->dump(); 
            return true;
        }
        compose_thread_id_tracking_map[val] = false;
        return false;
    }

    bool is_composed_by_block_id (Value* val)
    {
        /* the value may be calculated by several binary operations 
           but eventually there sho*/
        if (compose_block_id_tracking_map.find(val) != compose_block_id_tracking_map.end()) {
            return compose_block_id_tracking_map[val];
        }
           
        errs() << "===== " << __func__ << " ";
        val->dump();

        if (auto bin_inst = dyn_cast<BinaryOperator>(val)) {
            Value* op1 = bin_inst->getOperand(0);
            Value* op2 = bin_inst->getOperand(1);
            return is_composed_by_block_id(op1) || is_composed_by_block_id(op2); 
        }
        else if (is_1d_block_id(val)) {
            compose_block_id_tracking_map[val] = true;
            return true;
        }
        compose_block_id_tracking_map[val] = false;
        return false;
    }


    bool is_same_source (Value *val1, Value* val2) 
    {
        if (val1 == val2) {
            return true;
        }

        return false;
    }

    void test(Value* val, DataLayout& dl) 
    {
        errs() << "===== " << __func__ << ": ";
        val->dump();
        if (auto strip_ptr_cast_val = val->stripPointerCasts()) {
            errs() << "strip ptr cast: ";
            strip_ptr_cast_val->dump();
        }
        APInt apint(64, 12, false);
        if (auto strip_acc_cnst_offset_val = val->stripAndAccumulateConstantOffsets(dl, apint, false)) {
            errs() << "strip acc cnst offset: ";
            strip_acc_cnst_offset_val->dump();
        }
    }

    void mem_loc_test(StoreInst* store_inst)
    {
        MemoryLocation mem_loc = MemoryLocation::get(store_inst);
        //errs() << "=====> " << __func__ << ": ";
        //mem_loc.Ptr->dump();
    }

    void alias_test (AAResults& AA, StoreInst* store_inst)
    {
        BasicBlock* bb = store_inst->getParent();
        auto next_inst = bb->begin();
        while (&*next_inst != store_inst) next_inst++;
        //for (auto succ_bb : successors(bb)) {
        //    succ_bb->dump();
            next_inst++;
            for ( ; next_inst != bb->end(); next_inst++) {
                if (auto next_store_inst = dyn_cast<StoreInst>(&*next_inst)) {
                    AAQueryInfo aa_query = AAQueryInfo();
                    //if (auto alias_res = AA.alias(MemoryLocation::get(store_inst), MemoryLocation::get(next_store_inst), aa_query) 
                    
                    if (auto alias_res = AA.alias(store_inst->getPointerOperand(), 
                                                  LocationSize(4),
                                                  next_store_inst->getPointerOperand(),
                                                  LocationSize(4)
                                                  ) 
                    
                            != AliasResult::NoAlias ) {
                        errs() << " === Alias === : " << alias_res << "\n";
                        store_inst->dump(); store_inst->getPointerOperand()->dump();
                        next_store_inst->dump(); next_store_inst->getPointerOperand()->dump();
                    }
                    else {
                        errs() << "No alias\n";
                    }
                }
            }
        //}
    }
    
    void load_alias_test (AAResults& AA, LoadInst* load_inst)
    {
        BasicBlock* bb = load_inst->getParent();
        auto next_inst = bb->begin();
        while (&*next_inst != load_inst) next_inst++;
        //for (auto succ_bb : successors(bb)) {
        //    succ_bb->dump();
            next_inst++;
            for ( ; next_inst != bb->end(); next_inst++) {
                if (auto next_load_inst = dyn_cast<LoadInst>(&*next_inst)) {
                    AAQueryInfo aa_query = AAQueryInfo();
                    //if (auto alias_res = AA.alias(MemoryLocation::get(load_inst), MemoryLocation::get(next_load_inst), aa_query) 
                    
                    if (auto alias_res = AA.alias(load_inst->getPointerOperand(), 
                                                  next_load_inst->getPointerOperand()
                                                  ) 
                    
                            != AliasResult::NoAlias ) {
                        errs() << " === Alias === : " << alias_res << "\n";
                        load_inst->dump(); load_inst->getPointerOperand()->dump();
                        next_load_inst->dump(); next_load_inst->getPointerOperand()->dump();
                    }
                    else {
                        errs() << "No alias\n";
                    }
                }
            }
        //}
    }

    void aa_test(Value* val) 
    {   
    }

};

#endif
